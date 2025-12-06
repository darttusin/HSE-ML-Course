import sys
import streamlit as st
import pickle


from config import settings
from tabs.eda import eda
from tabs.file_predict import file_predict
from tabs.single_predict import single_predict_tab

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

st.title("🚗 Car Price Prediction App")

try:
    with open(settings.scaler_model_path, "rb") as f:
        scaler = pickle.load(f)

    with open(settings.ridge_model_path, "rb") as f:
        model = pickle.load(f)

    with open(settings.onehot_encoder_path, "rb") as f:
        onehot_encoder = pickle.load(f)

except FileNotFoundError as ex:
    st.error(f"Ошибка загрузки модели: {ex}")
    sys.exit(1)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Одиночное предсказание", "Массовое предсказание", "EDA", "Анализ модели"]
)

with tab1:
    single_predict_tab(model, onehot_encoder)

with tab2:
    file_predict(model, onehot_encoder)

with tab3:
    eda()

with tab4:
    st.header("Анализ модели")
    st.write("Здесь будет анализ важности признаков модели...")
