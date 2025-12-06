import streamlit as st

import pandas as pd
import numpy as np
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def extract(value):
    if pd.isna(value):
        return np.nan

    try:
        return float(str(value).split(" ")[0])
    except:
        return np.nan


def eda():
    st.header("📊 Exploratory Data Analysis (EDA)")

    st.markdown("""
    ### Анализ загруженных данных

    Загрузите CSV файл с данными об автомобилях для проведения анализа.
    """)

    uploaded_file_eda = st.file_uploader(
        "Выберите CSV файл для анализа", type=["csv"], key="eda_uploader"
    )

    if uploaded_file_eda is not None:
        try:
            df_eda = pd.read_csv(uploaded_file_eda)

            if "mileage" in df_eda.columns:
                df_eda["mileage"] = df_eda["mileage"].apply(extract)

            if "engine" in df_eda.columns:
                df_eda["engine"] = df_eda["engine"].apply(extract)

            if "max_power" in df_eda.columns:
                df_eda["max_power"] = df_eda["max_power"].apply(extract)

            st.subheader("📋 Общая информация о данных")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Количество записей", df_eda.shape[0])
            with col2:
                st.metric("Количество признаков", df_eda.shape[1])
            with col3:
                st.metric("Пропущенные значения", df_eda.isnull().sum().sum())

            with st.expander("👀 Просмотр данных (первые 10 строк)"):
                st.dataframe(df_eda.head(10), use_container_width=True)

            with st.expander("🔍 Информация о типах данных"):
                buffer = StringIO()
                df_eda.info(buf=buffer)
                st.text(buffer.getvalue())

            st.subheader("📊 Описательная статистика")

            numeric_cols = df_eda.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                with st.expander("Числовые признаки"):
                    st.dataframe(
                        df_eda[numeric_cols].describe(), use_container_width=True
                    )

            categorical_cols = df_eda.select_dtypes(include=["object"]).columns
            if len(categorical_cols) > 0:
                with st.expander("Категориальные признаки"):
                    for col in categorical_cols:
                        st.write(f"**{col}**:")
                        col_stats = df_eda[col].value_counts()
                        st.dataframe(col_stats, use_container_width=True)

            st.subheader("📈 Визуализация данных")

            viz_type = st.selectbox(
                "Выберите тип визуализации",
                [
                    "Распределение числовых признаков",
                    "Корреляционная матрица",
                    "Распределение категориальных признаков",
                    "Парные зависимости",
                ],
            )

            if viz_type == "Распределение числовых признаков":
                if len(numeric_cols) > 0:
                    selected_num = st.selectbox(
                        "Выберите числовой признак", numeric_cols
                    )

                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=("Гистограмма", "Box plot"),
                        specs=[[{"type": "histogram"}, {"type": "box"}]],
                    )

                    fig.add_trace(
                        go.Histogram(x=df_eda[selected_num], name="Распределение"),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Box(y=df_eda[selected_num], name="Box plot"), row=1, col=2
                    )

                    fig.update_layout(
                        title_text=f"Распределение признака {selected_num}", height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Среднее", f"{df_eda[selected_num].mean():.2f}")
                    with col2:
                        st.metric("Медиана", f"{df_eda[selected_num].median():.2f}")
                    with col3:
                        st.metric(
                            "Стандартное отклонение",
                            f"{df_eda[selected_num].std():.2f}",
                        )
                    with col4:
                        st.metric(
                            "Количество уникальных", df_eda[selected_num].nunique()
                        )

            elif viz_type == "Корреляционная матрица":
                if len(numeric_cols) > 1:
                    corr_matrix = df_eda[numeric_cols].corr()

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.index,
                            colorscale="RdBu",
                            zmin=-1,
                            zmax=1,
                            text=corr_matrix.round(2).values,
                            texttemplate="%{text}",
                            textfont={"size": 10},
                            hoverongaps=False,
                        )
                    )

                    fig.update_layout(
                        title="Корреляционная матрица числовых признаков", height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("**Самые сильные корреляции:**")
                    corr_pairs = corr_matrix.unstack()
                    sorted_pairs = corr_pairs.sort_values(key=abs, ascending=False)

                    unique_pairs = sorted_pairs[
                        sorted_pairs.index.get_level_values(0)
                        != sorted_pairs.index.get_level_values(1)
                    ]
                    unique_pairs = unique_pairs[~unique_pairs.index.duplicated()]

                    top_corrs = unique_pairs.head(5)

                    for (feature1, feature2), value in top_corrs.items():
                        st.write(f"{feature1} - {feature2}: **{value:.3f}**")

            elif viz_type == "Распределение категориальных признаков":
                if len(categorical_cols) > 0:
                    selected_cat = st.selectbox(
                        "Выберите категориальный признак", categorical_cols
                    )

                    value_counts = df_eda[selected_cat].value_counts().head(10)

                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=("Bar plot", "Pie chart"),
                        specs=[[{"type": "bar"}, {"type": "pie"}]],
                    )

                    fig.add_trace(
                        go.Bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            name="Количество",
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Pie(
                            labels=value_counts.index,
                            values=value_counts.values,
                            name="Доли",
                        ),
                        row=1,
                        col=2,
                    )

                    fig.update_layout(
                        title_text=f"Распределение признака {selected_cat}", height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write(
                        f"**Всего уникальных значений:** {df_eda[selected_cat].nunique()}"
                    )
                    st.write(
                        f"**Самое частое значение:** {df_eda[selected_cat].mode().iloc[0]}"
                    )

            elif viz_type == "Парные зависимости":
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_feature = st.selectbox(
                            "Признак X", numeric_cols, key="x_feature"
                        )
                    with col2:
                        y_feature = st.selectbox(
                            "Признак Y", numeric_cols, key="y_feature"
                        )

                    color_feature = None
                    if len(categorical_cols) > 0:
                        color_feature = st.selectbox(
                            "Цветовая группировка (опционально)",
                            ["Нет"] + list(categorical_cols),
                        )
                        if color_feature == "Нет":
                            color_feature = None

                    if x_feature and y_feature:
                        if color_feature:
                            fig = px.scatter(
                                df_eda,
                                x=x_feature,
                                y=y_feature,
                                color=color_feature,
                                title=f"Зависимость {y_feature} от {x_feature}",
                                hover_data=df_eda.columns,
                            )
                        else:
                            fig = px.scatter(
                                df_eda,
                                x=x_feature,
                                y=y_feature,
                                title=f"Зависимость {y_feature} от {x_feature}",
                                hover_data=df_eda.columns,
                            )

                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔍 Анализ пропущенных значений")

            missing_values = df_eda.isnull().sum()
            missing_percent = (missing_values / len(df_eda)) * 100

            missing_df = pd.DataFrame(
                {
                    "Количество пропусков": missing_values,
                    "Процент пропусков": missing_percent,
                }
            ).sort_values("Количество пропусков", ascending=False)

            missing_df = missing_df[missing_df["Количество пропусков"] > 0]

            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=missing_df.index,
                            y=missing_df["Процент пропусков"],
                            text=missing_df["Процент пропусков"].round(2),
                            textposition="auto",
                        )
                    ]
                )

                fig.update_layout(
                    title="Процент пропущенных значений по признакам",
                    xaxis_title="Признаки",
                    yaxis_title="Процент пропусков",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ В данных нет пропущенных значений!")

            st.subheader("📊 Выбросы в числовых признаках")

            if len(numeric_cols) > 0:
                selected_outlier = st.selectbox(
                    "Выберите признак для анализа выбросов",
                    numeric_cols,
                    key="outlier_select",
                )

                if selected_outlier:
                    Q1 = df_eda[selected_outlier].quantile(0.25)
                    Q3 = df_eda[selected_outlier].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = df_eda[
                        (df_eda[selected_outlier] < lower_bound)
                        | (df_eda[selected_outlier] > upper_bound)
                    ]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Выбросов", len(outliers))
                    with col2:
                        st.metric(
                            "Процент выбросов",
                            f"{(len(outliers) / len(df_eda) * 100):.2f}%",
                        )
                    with col3:
                        st.metric("Нижняя граница", f"{lower_bound:.2f}")
                    with col4:
                        st.metric("Верхняя граница", f"{upper_bound:.2f}")

                    fig = go.Figure()
                    fig.add_trace(
                        go.Box(y=df_eda[selected_outlier], name=selected_outlier)
                    )

                    fig.update_layout(
                        title=f"Выбросы в признаке {selected_outlier}", height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("📥 Экспорт анализа")

            analysis_results = {
                "Общая информация": {
                    "Количество записей": df_eda.shape[0],
                    "Количество признаков": df_eda.shape[1],
                    "Пропущенные значения": int(df_eda.isnull().sum().sum()),
                    "Дубликаты": int(df_eda.duplicated().sum()),
                },
                "Числовые признаки": df_eda[numeric_cols].describe().to_dict()
                if len(numeric_cols) > 0
                else {},
                "Категориальные признаки": {
                    col: {
                        "Количество уникальных": int(df_eda[col].nunique()),
                        "Самое частое": str(df_eda[col].mode().iloc[0])
                        if len(df_eda[col].mode()) > 0
                        else "Нет",
                    }
                    for col in categorical_cols
                },
            }

            if st.button("📊 Экспортировать анализ в JSON"):
                import json

                analysis_json = json.dumps(
                    analysis_results, indent=2, ensure_ascii=False
                )
                st.download_button(
                    label="Скачать JSON",
                    data=analysis_json,
                    file_name="eda_analysis.json",
                    mime="application/json",
                )

        except Exception as e:
            st.error(f"Ошибка при анализе данных: {str(e)}")
            st.error(
                "Пожалуйста, проверьте формат файла и наличие необходимых колонок."
            )

    else:
        st.info("👆 Загрузите CSV файл для начала анализа")

        with st.expander("📋 Пример структуры файла"):
            example_data = pd.DataFrame(
                {
                    "year": [2015, 2017, 2018, 2019, 2020],
                    "km_driven": [50000, 30000, 70000, 25000, 40000],
                    "mileage": [15.0, 18.0, 12.0, 20.0, 16.0],
                    "engine": [1200, 1500, 1000, 1300, 1400],
                    "max_power": [80.0, 90.0, 70.0, 85.0, 95.0],
                    "seats": [5, 5, 4, 5, 5],
                    "fuel": ["Petrol", "Diesel", "Petrol", "Petrol", "Diesel"],
                    "seller_type": [
                        "Individual",
                        "Dealer",
                        "Individual",
                        "Dealer",
                        "Individual",
                    ],
                    "transmission": [
                        "Manual",
                        "Automatic",
                        "Manual",
                        "Manual",
                        "Automatic",
                    ],
                    "owner": [
                        "First Owner",
                        "Second Owner",
                        "First Owner",
                        "First Owner",
                        "Second Owner",
                    ],
                }
            )
            st.dataframe(example_data, use_container_width=True)
