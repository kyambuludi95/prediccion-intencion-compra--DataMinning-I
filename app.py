"""App Streamlit: predicción de intención de compra (Online Shoppers Intention).

Carga los modelos entrenados en notebooks/Clase3/trabajo_g03.ipynb
(KNN serializado con joblib, MLP serializado con Keras) y permite
predecir en tiempo real, tanto por formulario como por carga de CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib

BASE_DIR = Path(__file__).parent
KNN_PATH = BASE_DIR / "models" / "knn_trabajo_g03.joblib"
MLP_PATH = BASE_DIR / "models" / "mlp_trabajo_g03.keras"
DATA_PATH = BASE_DIR / "data" / "online_shoppers_intention.csv"
#KNN_PATH = BASE_DIR / "output" / "models" / "knn_trabajo_g03.joblib"
#MLP_PATH = BASE_DIR / "output" / "models" / "mlp_trabajo_g03.keras"
#DATA_PATH = BASE_DIR / "data" / "raw" / "online_shoppers_intention.csv"

FEATURE_COLUMNS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]
NUMERIC_COLUMNS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
# El OneHotEncoder del pipeline fue fiteado con estas columnas como string.
# Si llegan como int, handle_unknown='ignore' las convierte en dummies
# todo-cero SIN lanzar error: la predicción se degrada en silencio.
STRING_CODED_COLUMNS = ["OperatingSystems", "Browser", "Region", "TrafficType"]


@st.cache_resource
def load_models():
    if not KNN_PATH.exists() or not MLP_PATH.exists():
        st.error(
            f"No se encontraron los artefactos del modelo. "
            f"Esperados en {KNN_PATH} y {MLP_PATH}."
        )
        st.stop()
    knn_pipeline = joblib.load(KNN_PATH)
    mlp_model = tf.keras.models.load_model(MLP_PATH)
    # ADVERTENCIA: el MLP no tiene su propio preprocesador serializado.
    # Reutilizamos el ColumnTransformer embebido en el pipeline del KNN
    # porque ambos se entrenaron sobre el mismo split (random_state=42).
    # Si el KNN se reentrena con otro split, este acoplamiento se rompe
    # EN SILENCIO (el MLP seguirá prediciendo, pero con datos mal escalados).
    preprocessor = knn_pipeline.named_steps["prep"]
    return knn_pipeline, mlp_model, preprocessor


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    return df


def build_input_frame(values: dict) -> pd.DataFrame:
    """Construye el DataFrame de 17 columnas que espera el pipeline."""
    df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    for col in STRING_CODED_COLUMNS:
        df[col] = df[col].astype(str)
    df["Weekend"] = df["Weekend"].astype(int)
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype(float)
    return df


def validate_batch_schema(df: pd.DataFrame) -> tuple[bool, str]:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Faltan columnas requeridas: {', '.join(missing)}"
    return True, ""


def predict_knn(knn_pipeline, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    preds = knn_pipeline.predict(df)
    probas = knn_pipeline.predict_proba(df)[:, 1]
    return preds, probas


def predict_mlp(mlp_model, preprocessor, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = preprocessor.transform(df)
    probas = mlp_model.predict(X, verbose=0).reshape(-1)
    preds = (probas >= 0.5).astype(int)
    return preds, probas


def render_prediction(label: str, pred: int, proba: float):
    resultado = "Compra" if pred == 1 else "No compra"
    st.metric(label=f"{label}: {resultado}", value=f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))


def render_form(knn_pipeline, mlp_model, preprocessor, ref_df: pd.DataFrame):
    months = sorted(ref_df["Month"].unique().tolist())
    visitor_types = sorted(ref_df["VisitorType"].unique().tolist())
    os_options = sorted(ref_df["OperatingSystems"].unique().tolist())
    browser_options = sorted(ref_df["Browser"].unique().tolist())
    region_options = sorted(ref_df["Region"].unique().tolist())
    traffic_options = sorted(ref_df["TrafficType"].unique().tolist())

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Actividad administrativa")
            administrative = st.number_input("Administrative (páginas)", min_value=0, value=0)
            administrative_duration = st.number_input(
                "Administrative_Duration (seg)", min_value=0.0, value=0.0
            )
            informational = st.number_input("Informational (páginas)", min_value=0, value=0)
            informational_duration = st.number_input(
                "Informational_Duration (seg)", min_value=0.0, value=0.0
            )

        with col2:
            st.subheader("Actividad de producto")
            product_related = st.number_input("ProductRelated (páginas)", min_value=0, value=1)
            product_related_duration = st.number_input(
                "ProductRelated_Duration (seg)", min_value=0.0, value=30.0
            )
            bounce_rates = st.number_input(
                "BounceRates", min_value=0.0, max_value=1.0, value=0.02, format="%.4f"
            )
            exit_rates = st.number_input(
                "ExitRates", min_value=0.0, max_value=1.0, value=0.04, format="%.4f"
            )

        with col3:
            st.subheader("Contexto de sesión")
            page_values = st.number_input("PageValues", min_value=0.0, value=0.0)
            special_day = st.number_input(
                "SpecialDay", min_value=0.0, max_value=1.0, value=0.0, format="%.1f"
            )
            month = st.selectbox("Month", months, index=months.index("May") if "May" in months else 0)
            visitor_type = st.selectbox("VisitorType", visitor_types)

        col4, col5, col6 = st.columns(3)
        with col4:
            operating_system = st.selectbox("OperatingSystems", os_options)
            browser = st.selectbox("Browser", browser_options)
        with col5:
            region = st.selectbox("Region", region_options)
            traffic_type = st.selectbox("TrafficType", traffic_options)
        with col6:
            weekend = st.checkbox("Weekend", value=False)

        submitted = st.form_submit_button("Predecir")

    if submitted:
        values = {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend,
        }
        input_df = build_input_frame(values)

        knn_pred, knn_proba = predict_knn(knn_pipeline, input_df)
        mlp_pred, mlp_proba = predict_mlp(mlp_model, preprocessor, input_df)

        st.divider()
        st.subheader("Resultado")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            render_prediction("KNN (k=51)", int(knn_pred[0]), float(knn_proba[0]))
        with res_col2:
            render_prediction("MLP", int(mlp_pred[0]), float(mlp_proba[0]))

        st.caption(
            "⚠️ El KNN se entrenó con RandomUnderSampler (distribución 50/50), "
            "no con la proporción real de la población (≈84.5% no compra / 15.5% compra). "
            "Las probabilidades mostradas sobreestiman la propensión de compra real. "
            "Además, con k=51 y weights='uniform', predict_proba solo puede tomar "
            "múltiplos de 1/51 (≈0.0196)."
        )


def render_batch(knn_pipeline, mlp_model, preprocessor):
    st.subheader("Carga de CSV por lotes")
    st.write(
        f"El archivo debe contener las columnas: `{', '.join(FEATURE_COLUMNS)}`. "
        "Columnas adicionales (por ejemplo `Revenue`) se ignoran."
    )
    uploaded = st.file_uploader("Selecciona un CSV", type="csv")

    if uploaded is None:
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        return

    batch_df = batch_df.drop(columns=[c for c in batch_df.columns if c.startswith("Unnamed")], errors="ignore")

    is_valid, error_msg = validate_batch_schema(batch_df)
    if not is_valid:
        st.error(error_msg)
        return

    MAX_ROWS = 10_000
    if len(batch_df) > MAX_ROWS:
        st.warning(f"El archivo tiene {len(batch_df)} filas; se procesarán solo las primeras {MAX_ROWS}.")
        batch_df = batch_df.head(MAX_ROWS)

    working_df = batch_df[FEATURE_COLUMNS].copy()
    for col in STRING_CODED_COLUMNS:
        working_df[col] = working_df[col].astype(str)
    working_df["Weekend"] = working_df["Weekend"].astype(int)
    for col in NUMERIC_COLUMNS:
        working_df[col] = working_df[col].astype(float)

    knn_preds, knn_probas = predict_knn(knn_pipeline, working_df)
    mlp_preds, mlp_probas = predict_mlp(mlp_model, preprocessor, working_df)

    result_df = batch_df.copy()
    result_df["pred_knn"] = knn_preds
    result_df["proba_knn"] = knn_probas
    result_df["pred_mlp"] = mlp_preds
    result_df["proba_mlp"] = mlp_probas

    st.success(f"Se generaron predicciones para {len(result_df)} filas.")
    st.dataframe(result_df, use_container_width=True)

    st.download_button(
        "Descargar resultados (CSV)",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="predicciones.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(
        page_title="Predicción de Intención de Compra",
        layout="wide",
    )
    st.title("Predicción de intención de compra online")

    knn_pipeline, mlp_model, preprocessor = load_models()
    ref_df = load_reference_data()

    with st.sidebar:
        st.header("Acerca de")
        st.write(
            "Predice si una sesión de navegación en un sitio de e-commerce "
            "terminará en compra (`Revenue`), a partir de datos de la sesión "
            "(dataset Online Shoppers Intention, UCI)."
        )
        st.subheader("Modelos")
        st.write("- **KNN**: k=51, weights='uniform', undersampling (RandomUnderSampler)")
        st.write("- **MLP**: red neuronal Keras, 74 entradas tras preprocesamiento")
        st.subheader("Preprocesamiento")
        st.write(
            "OneHotEncoder para variables nominales, StandardScaler para "
            "variables numéricas, `Weekend` como passthrough."
        )

    tab_individual, tab_batch = st.tabs(["Predicción individual", "Predicción por lotes"])

    with tab_individual:
        render_form(knn_pipeline, mlp_model, preprocessor, ref_df)

    with tab_batch:
        render_batch(knn_pipeline, mlp_model, preprocessor)


if __name__ == "__main__":
    main()
