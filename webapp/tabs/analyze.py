import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from config import settings
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def analyze(model, onehot_encoder):
    st.header("🔍 Анализ модели")

    st.markdown("""
    ### Анализ работы модели Ridge Regression

    На этой вкладке вы можете:
    1. Посмотреть важность признаков
    2. Проанализировать метрики модели
    3. Исследовать SHAP значения
    4. Загрузить тестовые данные для оценки
    """)

    st.subheader("📊 Загрузка тестовых данных")
    test_file = st.file_uploader(
        "Загрузите тестовые данные с целевой переменной 'price'",
        type=["csv"],
        key="test_uploader",
    )

    if test_file is not None:
        try:
            df_test = pd.read_csv(test_file)

            if "selling_price" not in df_test.columns:
                st.error("❌ В данных отсутствует колонка 'price' - целевая переменная")
            else:
                y_test = df_test["selling_price"]
                X_test = df_test.drop("selling_price", axis=1)

                required_features = (
                    settings.numeric_features + settings.categorical_features
                )
                missing_features = [
                    f for f in required_features if f not in X_test.columns
                ]

                if missing_features:
                    st.error(f"❌ Отсутствуют признаки: {', '.join(missing_features)}")
                else:
                    with st.spinner("Обработка тестовых данных..."):
                        numeric_df = X_test[settings.numeric_features]

                        categorical_encoded = onehot_encoder.transform(
                            X_test[settings.categorical_features]
                        )
                        feature_names = onehot_encoder.get_feature_names_out(
                            settings.categorical_features
                        )
                        categorical_df = pd.DataFrame(
                            categorical_encoded, columns=feature_names
                        )

                        X_final = pd.concat([numeric_df, categorical_df], axis=1)
                        y_pred = model.predict(X_final)

                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        st.subheader("📈 Метрики модели")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.2f}")
                        with col3:
                            st.metric("MAE", f"{mae:.2f}")
                        with col4:
                            st.metric("MSE", f"{mse:.2f}")

                        st.subheader(
                            "📊 Сравнение предсказаний с фактическими значениями"
                        )

                        fig = make_subplots(
                            rows=1,
                            cols=2,
                            subplot_titles=(
                                "Предсказания vs Факт",
                                "Ошибки предсказаний",
                            ),
                            specs=[[{"type": "scatter"}, {"type": "histogram"}]],
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=y_test,
                                y=y_pred,
                                mode="markers",
                                marker=dict(size=8, opacity=0.6),
                                name="Предсказания",
                                text=[
                                    f"Факт: {fact:.0f}, Предск.: {pred:.0f}"
                                    for fact, pred in zip(y_test, y_pred)
                                ],
                            ),
                            row=1,
                            col=1,
                        )

                        # Идеальная линия
                        max_val = max(y_test.max(), y_pred.max())
                        min_val = min(y_test.min(), y_pred.min())
                        fig.add_trace(
                            go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode="lines",
                                line=dict(color="red", dash="dash"),
                                name="Идеальная линия",
                            ),
                            row=1,
                            col=1,
                        )

                        errors = y_pred - y_test
                        fig.add_trace(
                            go.Histogram(
                                x=errors,
                                nbinsx=50,
                                name="Распределение ошибок",
                                marker_color="lightblue",
                            ),
                            row=1,
                            col=2,
                        )

                        fig.update_xaxes(title_text="Фактическая цена", row=1, col=1)
                        fig.update_yaxes(title_text="Предсказанная цена", row=1, col=1)
                        fig.update_xaxes(
                            title_text="Ошибка (Предсказание - Факт)", row=1, col=2
                        )
                        fig.update_yaxes(title_text="Количество", row=1, col=2)

                        fig.update_layout(height=500, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

                        st.write("**Статистика ошибок:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Средняя ошибка", f"{errors.mean():.2f}")
                        with col2:
                            st.metric("Стандартное отклонение", f"{errors.std():.2f}")
                        with col3:
                            st.metric("Медианная ошибка", f"{errors.median():.2f}")
                        with col4:
                            st.metric(
                                "Максимальная ошибка", f"{errors.abs().max():.2f}"
                            )

        except Exception as e:
            st.error(f"Ошибка при обработке тестовых данных: {str(e)}")

    st.subheader("🎯 Важность признаков")

    if hasattr(model, "coef_"):
        coefficients = model.coef_
        feature_names_final = X_final.columns if "X_final" in locals() else []

        if len(feature_names_final) > 0:
            feature_importance = pd.DataFrame(
                {
                    "Признак": feature_names_final,
                    "Коэффициент": coefficients,
                    "Абсолютное значение": np.abs(coefficients),
                }
            ).sort_values("Абсолютное значение", ascending=False)

            fig = px.bar(
                feature_importance.head(20),
                x="Абсолютное значение",
                y="Признак",
                orientation="h",
                title="Топ-20 самых важных признаков",
                color="Коэффициент",
                color_continuous_scale="RdBu",
                labels={"Абсолютное значение": "Абс. значение коэффициента"},
            )

            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Детальная таблица важности признаков"):
                st.dataframe(
                    feature_importance.style.background_gradient(
                        subset=["Абсолютное значение"], cmap="Blues"
                    ),
                    use_container_width=True,
                )

        else:
            st.info("Загрузите тестовые данные для анализа важности признаков")
    else:
        st.warning("Модель не имеет атрибута 'coef_' для анализа важности признаков")

    st.subheader("ℹ️ Информация о модели")
    st.write("**Параметры модели:**")
    model_params = model.get_params() if hasattr(model, "get_params") else {}

    params_df = pd.DataFrame(
        {
            "Параметр": list(model_params.keys()),
            "Значение": list(model_params.values()),
        }
    )
    st.dataframe(params_df, use_container_width=True)
