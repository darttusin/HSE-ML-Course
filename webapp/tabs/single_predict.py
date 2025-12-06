import streamlit as st
import pandas as pd
from config import settings


def single_predict_tab(model, onehot_encoder):
    st.header("Предсказание цены для одного автомобиля")

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input(
                "Год выпуска", min_value=1990, max_value=2024, value=2015
            )
            km_driven = st.number_input(
                "Пробег (км)", min_value=0, max_value=1000000, value=50000
            )
            mileage = st.number_input(
                "Расход топлива (км/л)", min_value=0.0, max_value=50.0, value=15.0
            )

        with col2:
            engine = st.number_input(
                "Объем двигателя (cc)", min_value=0, max_value=5000, value=1200
            )
            max_power = st.number_input(
                "Мощность (bhp)", min_value=0.0, max_value=500.0, value=80.0
            )
            seats = st.number_input(
                "Количество мест", min_value=2, max_value=10, value=5
            )

        col3, col4 = st.columns(2)

        with col3:
            fuel = st.selectbox("Тип топлива", ["Diesel", "Petrol", "LPG", "CNG"])
            seller_type = st.selectbox(
                "Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"]
            )

        with col4:
            transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
            owner = st.selectbox(
                "Владелец",
                ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"],
            )

        submitted = st.form_submit_button("Предсказать цену")

    if submitted:
        numeric_data = pd.DataFrame(
            [[year, km_driven, mileage, engine, max_power]],
            columns=settings.numeric_features,  # type: ignore
        )
        categorical_data = pd.DataFrame(
            [[fuel, seller_type, transmission, owner, seats]],
            columns=settings.categorical_features,  # type: ignore
        )
        categorical_encoded = onehot_encoder.transform(categorical_data)

        feature_names = onehot_encoder.get_feature_names_out(
            settings.categorical_features
        )
        categorical_df = pd.DataFrame(categorical_encoded, columns=feature_names)

        X_final = pd.concat([numeric_data, categorical_df], axis=1)

        prediction = model.predict(X_final)

        st.success(f"Предсказанная цена автомобиля: **{prediction[0]:,.2f}**")

        with st.expander("Детали предсказания"):
            st.write("**Введенные параметры:**")
            st.json(
                {
                    "year": year,
                    "km_driven": km_driven,
                    "mileage": mileage,
                    "engine": engine,
                    "max_power": max_power,
                    "seats": seats,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner,
                }
            )
