import streamlit as st
import pandas as pd
import base64
from config import settings


def file_predict(model, onehot_encoder):
    st.header("Массовое предсказание по файлу")

    st.markdown("""
    ### Загрузите CSV файл с данными автомобилей

    Файл должен содержать следующие колонки:
    - `year`: Год выпуска
    - `km_driven`: Пробег (км)
    - `mileage`: Расход топлива (км/л)
    - `engine`: Объем двигателя (cc)
    - `max_power`: Мощность (bhp)
    - `seats`: Количество мест
    - `fuel`: Тип топлива (Diesel/Petrol/LPG/CNG)
    - `seller_type`: Тип продавца (Individual/Dealer/Trustmark Dealer)
    - `transmission`: Коробка передач (Manual/Automatic)
    - `owner`: Владелец (First Owner/Second Owner/Third Owner/Fourth & Above Owner)
    """)

    uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("Предпросмотр данных")
            st.dataframe(df.head(), use_container_width=True)

            required_columns = settings.numeric_features.copy()
            required_columns.extend(settings.categorical_features)

            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"Отсутствуют следующие колонки: {', '.join(missing_columns)}")
            else:
                if st.button("Выполнить предсказания", type="primary"):
                    with st.spinner("Обработка данных..."):
                        numeric_df = df[settings.numeric_features]

                        categorical_encoded = onehot_encoder.transform(
                            df[settings.categorical_features]
                        )
                        feature_names = onehot_encoder.get_feature_names_out(
                            settings.categorical_features
                        )
                        categorical_df = pd.DataFrame(
                            categorical_encoded, columns=feature_names
                        )

                        X_final = pd.concat([numeric_df, categorical_df], axis=1)

                        predictions = model.predict(X_final)

                        df_result = df.copy()
                        df_result["predicted_price"] = predictions

                        st.subheader("Результаты предсказаний")
                        st.dataframe(
                            df_result[required_columns + ["predicted_price"]].head(),
                            use_container_width=True,
                        )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Количество записей", len(df_result))
                        with col2:
                            st.metric(
                                "Средняя цена",
                                f"{df_result['predicted_price'].mean():,.2f}",
                            )
                        with col3:
                            st.metric(
                                "Максимальная цена",
                                f"{df_result['predicted_price'].max():,.2f}",
                            )

                        st.subheader("Скачать результаты")

                        csv = df_result.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()

                        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Скачать CSV файл с результатами</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        st.subheader("Распределение предсказанных цен")
                        st.bar_chart(df_result["predicted_price"])

        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")

    with st.expander("Пример структуры CSV файла"):
        sample_data = pd.DataFrame(
            {
                "year": [2015, 2017, 2018],
                "km_driven": [50000, 30000, 70000],
                "mileage": [15.0, 18.0, 12.0],
                "engine": [1200, 1500, 1000],
                "max_power": [80.0, 90.0, 70.0],
                "seats": [5, 5, 4],
                "fuel": ["Petrol", "Diesel", "Petrol"],
                "seller_type": ["Individual", "Dealer", "Individual"],
                "transmission": ["Manual", "Automatic", "Manual"],
                "owner": ["First Owner", "Second Owner", "First Owner"],
            }
        )
        st.dataframe(sample_data)
