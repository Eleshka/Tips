import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="Исследование чаевых",
    layout="wide"
)

# Заголовок приложения
st.title("Исследование датасета чаевых")
st.markdown("---")

# Боковая панель для загрузки файла и настроек
with st.sidebar:
    st.header("Загрузка данных")
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Загрузите CSV файл", 
        type=['csv'],
        help="Загрузите файл tips.csv или аналогичный датасет"
    )
    
    st.markdown("---")
    st.header("Настройки визуализации")
    
    # Выбор темы для графиков
    theme = st.selectbox(
        "Тема графика",
        ["whitegrid", "darkgrid", "white", "dark"]
    )
    
    # Цветовая палитра
    palette = st.selectbox(
        "Цветовая палитра",
        ["viridis", "plasma", "coolwarm", "Set2", "pastel"]
    )
    
    st.markdown("---")
    st.header("Фильтры данных")
    
    # Фильтры (будут активированы после загрузки данных)
    day_filter = st.multiselect(
        "День недели",
        ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"],
        default=["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"]
    )
    
    time_filter = st.multiselect(
        "Время дня",
        ["Lunch", "Dinner"],
        default=["Lunch", "Dinner"]
    )

# Функция для загрузки данных по умолчанию
@st.cache_data
def load_default_data():
    try:
        # Загрузка данных по ссылке
        path = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
        tips = pd.read_csv(path)
        
        # Создаем столбец time_order со случайными датами
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        # Генерируем случайные даты
        tips['time_order'] = pd.to_datetime(np.random.choice(
            pd.date_range(start=start_date, end=end_date),
            size=len(tips)
        ))
        
        # Создаем английские сокращения дней недели
        day_short_en = {
            'Monday': 'Mon',
            'Tuesday': 'Tue', 
            'Wednesday': 'Wed',
            'Thursday': 'Thur',
            'Friday': 'Fri',
            'Saturday': 'Sat',
            'Sunday': 'Sun'
        }
        
        tips['day'] = tips['time_order'].dt.day_name().map(day_short_en)
        
        return tips
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        # Если не получается, создаем демо-данные
        return pd.DataFrame({
            'total_bill': [16.99, 10.34, 21.01, 23.68, 24.59],
            'tip': [1.01, 1.66, 3.50, 3.31, 3.61],
            'sex': ['Female', 'Male', 'Male', 'Male', 'Female'],
            'smoker': ['No', 'No', 'No', 'No', 'No'],
            'day': ['Sun', 'Sun', 'Sun', 'Sun', 'Sun'],
            'time': ['Dinner', 'Dinner', 'Dinner', 'Dinner', 'Dinner'],
            'size': [2, 3, 3, 2, 4]
        })
    
# Функция для преобразования DataFrame в Arrow-совместимый формат
def make_df_arrow_compatible(df):
    """Преобразует DataFrame для совместимости с Arrow"""
    df_copy = df.copy()
    
    # Преобразуем datetime столбцы в строки для отображения
    datetime_columns = df_copy.select_dtypes(include=['datetime64']).columns
    for col in datetime_columns:
        df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df_copy

# Основная логика приложения
if uploaded_file is not None:
    # Загружаем данные пользователя
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Файл успешно загружен!")
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        df = load_default_data()
else:
    # Используем данные по умолчанию
    df = load_default_data()
    st.sidebar.info("Используются демонстрационные данные из GitHub")

# Применяем фильтры
if 'day' in df.columns and 'time' in df.columns:
    filtered_df = df[
        (df['day'].isin(day_filter)) & 
        (df['time'].isin(time_filter))
    ]
else:
    filtered_df = df

# Создаем Arrow-совместимую версию для отображения
display_df = make_df_arrow_compatible(filtered_df)

# Настройка стиля графиков
sns.set_style(theme)

# Основная область приложения
tab1, tab2, tab3, tab4 = st.tabs(["Обзор данных", "Визуализация", "Анализ", "Экспорт"])

with tab1:
    st.header("Обзор данных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Основная информация")
        st.write(f"**Количество записей:** {len(filtered_df)}")
        st.write(f"**Количество столбцов:** {len(filtered_df.columns)}")
        
        # Основные статистики
        if 'total_bill' in filtered_df.columns:
            st.metric("Средний счет", f"${filtered_df['total_bill'].mean():.2f}")
        if 'tip' in filtered_df.columns:
            st.metric("Средние чаевые", f"${filtered_df['tip'].mean():.2f}")
        
        # Дополнительные метрики
        if all(col in filtered_df.columns for col in ['total_bill', 'tip']):
            tip_percentage = (filtered_df['tip'] / filtered_df['total_bill'] * 100).mean()
            st.metric("Средний % чаевых", f"{tip_percentage:.1f}%")
    
    with col2:
        st.subheader("Первые 10 записей")
        st.dataframe(display_df.head(10), width='stretch')
    
    st.subheader("Статистическое описание числовых данных")
    st.dataframe(filtered_df.describe(), width='stretch')

with tab2:
    st.header("Визуализация данных")
    
    # 1. Распределения
    st.subheader("Распределения")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Распределение счетов с KDE
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        if 'total_bill' in filtered_df.columns:
            sns.histplot(data=filtered_df, x='total_bill', kde=True, ax=ax1, 
                        color='skyblue', bins=20, alpha=0.7)
            ax1.set_title('Распределение сумм счетов')
            ax1.set_xlabel('Сумма счета ($)')
            ax1.set_ylabel('Частота')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            # Кнопка скачивания графика 1
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
            st.download_button(
                label="Скачать график распределения счетов",
                data=buf1.getvalue(),
                file_name="distribution_total_bill.png",
                mime="image/png",
                key="download_dist1"
            )
    
    with col2:
        # Распределение чаевых с KDE
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        if 'tip' in filtered_df.columns:
            sns.histplot(data=filtered_df, x='tip', kde=True, ax=ax2, 
                        color='lightcoral', bins=20, alpha=0.7)
            ax2.set_title('Распределение чаевых')
            ax2.set_xlabel('Чаевые ($)')
            ax2.set_ylabel('Частота')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
            # Кнопка скачивания графика 2
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
            st.download_button(
                label="Скачать график распределения чаевых",
                data=buf2.getvalue(),
                file_name="distribution_tips.png",
                mime="image/png",
                key="download_dist2"
            )
    
    # 2. Сравнительные распределения
    st.subheader("Сравнительные распределения")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Сравнение по курящим/некурящим
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        if all(col in filtered_df.columns for col in ['total_bill', 'smoker']):
            sns.kdeplot(data=filtered_df, x='total_bill', hue='smoker', 
                       fill=True, palette=palette, ax=ax3)
            ax3.set_title('Сравнение распределений по курящим/некурящим')
            ax3.set_xlabel('Сумма счета ($)')
            ax3.set_ylabel('Плотность')
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            
            # Кнопка скачивания графика 3
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format='png', dpi=300, bbox_inches='tight')
            st.download_button(
                label="Скачать график сравнения по курению",
                data=buf3.getvalue(),
                file_name="kde_smoker_comparison.png",
                mime="image/png",
                key="download_kde1"
            )
    
    with col4:
        # Сравнение по полу
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        if all(col in filtered_df.columns for col in ['total_bill', 'sex']):
            sns.kdeplot(data=filtered_df, x='total_bill', hue='sex', 
                       fill=True, palette=palette, ax=ax4)
            ax4.set_title('Сравнение распределений по полу')
            ax4.set_xlabel('Сумма счета ($)')
            ax4.set_ylabel('Плотность')
            ax4.grid(True, alpha=0.3)
            st.pyplot(fig4)
            
            # Кнопка скачивания графика 4
            buf4 = io.BytesIO()
            fig4.savefig(buf4, format='png', dpi=300, bbox_inches='tight')
            st.download_button(
                label="Скачать график сравнения по полу",
                data=buf4.getvalue(),
                file_name="kde_sex_comparison.png",
                mime="image/png",
                key="download_kde2"
            )
    
    # 3. Взаимосвязи
    st.subheader("Взаимосвязи")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Scatter plot с размером компании
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        if all(col in filtered_df.columns for col in ['total_bill', 'tip', 'size']):
            scatter = sns.scatterplot(data=filtered_df, x='total_bill', y='tip', 
                                    size='size', hue='size', sizes=(50, 300), 
                                    palette='viridis', alpha=0.7, ax=ax5)
            ax5.set_title('Связь: сумма счета vs чаевые vs размер компании')
            ax5.set_xlabel('Сумма счета ($)')
            ax5.set_ylabel('Чаевые ($)')
            ax5.grid(True, alpha=0.3)
            st.pyplot(fig5)
            
            # Кнопка скачивания графика 5
            buf5 = io.BytesIO()
            fig5.savefig(buf5, format='png', dpi=300, bbox_inches='tight')
            st.download_button(
                label="Скачать график зависимости чаевых от размера компании",
                data=buf5.getvalue(),
                file_name="scatter_size_plot.png",
                mime="image/png",
                key="download_scatter1"
            )
    
    with col6:
        # Scatter plot по полу и курению
        if all(col in filtered_df.columns for col in ['total_bill', 'tip', 'sex', 'smoker']):
            fig6, (ax6, ax7) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Мужчины
            male_data = filtered_df[filtered_df['sex'] == 'Male']
            sns.scatterplot(data=male_data, x='total_bill', y='tip', 
                          hue='smoker', style='smoker', 
                          palette={'Yes': 'red', 'No': 'blue'}, ax=ax6)
            ax6.set_title('Мужчины')
            ax6.set_xlabel('Сумма счета ($)')
            ax6.set_ylabel('Чаевые ($)')
            ax6.grid(True, alpha=0.3)
            
            # Женщины
            female_data = filtered_df[filtered_df['sex'] == 'Female']
            sns.scatterplot(data=female_data, x='total_bill', y='tip', 
                          hue='smoker', style='smoker', 
                          palette={'Yes': 'red', 'No': 'blue'}, ax=ax7)
            ax7.set_title('Женщины')
            ax7.set_xlabel('Сумма счета ($)')
            ax7.set_ylabel('Чаевые ($)')
            ax7.grid(True, alpha=0.3)
            
            plt.suptitle('Связь счета и чаевых с разбивкой по полу и курению', fontsize=14)
            st.pyplot(fig6)
            
            # Кнопка скачивания графика 6
            buf6 = io.BytesIO()
            fig6.savefig(buf6, format='png', dpi=300, bbox_inches='tight')
            st.download_button(
                label="Скачать график по полу и курению",
                data=buf6.getvalue(),
                file_name="scatter_sex_smoker.png",
                mime="image/png",
                key="download_scatter2"
            )
    
    # 4. Анализ по дням и времени
    st.subheader("Анализ по дням и времени")
    
    col7, col8 = st.columns(2)
    
    with col7:
        # Boxplot по дням недели
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        if all(col in filtered_df.columns for col in ['total_bill', 'day']):
            day_order = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
            sns.boxplot(data=filtered_df, x='day', y='total_bill', hue='day',
                       order=day_order, palette=palette, ax=ax7, legend=False)
            ax7.set_title('Распределение суммы счетов по дням недели')
            ax7.set_xlabel('День недели')
            ax7.set_ylabel('Сумма счета ($)')
            ax7.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig7)
            
            # Кнопка скачивания графика 7
            buf7 = io.BytesIO()
            fig7.savefig(buf7, format='png', dpi=300, bbox_inches='tight')
            st.download_button(
                label="Скачать box plot по дням",
                data=buf7.getvalue(),
                file_name="boxplot_days.png",
                mime="image/png",
                key="download_box1"
            )
    
    with col8:
        # Boxplot по дням и времени
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        if all(col in filtered_df.columns for col in ['total_bill', 'day', 'time']):
            day_order = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
            sns.boxplot(data=filtered_df, x='day', y='total_bill', hue='time',
                       order=day_order, palette=palette, ax=ax8)
            ax8.set_title('Распределение суммы счетов по дням и времени')
            ax8.set_xlabel('День недели')
            ax8.set_ylabel('Сумма счета ($)')
            ax8.legend(title='Время')
            ax8.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig8)
            
            # Кнопка скачивания графика 8
            buf8 = io.BytesIO()
            fig8.savefig(buf8, format='png', dpi=300, bbox_inches='tight')
            st.download_button(
                label="Скачать box plot по дням и времени",
                data=buf8.getvalue(),
                file_name="boxplot_days_time.png",
                mime="image/png",
                key="download_box2"
            )
    
    # 5. Динамика чаевых по времени
    st.subheader("Динамика чаевых во времени")
    
    if 'time_order' in filtered_df.columns and 'tip' in filtered_df.columns:
        # Сортируем данные по дате
        tips_sorted = filtered_df.sort_values('time_order')
        
        # Суммарные чаевые по дням
        daily_tips_sum = tips_sorted.groupby('time_order')['tip'].sum().reset_index()
        daily_tips_sum.columns = ['time_order', 'total_tips']
        
        fig9, ax9 = plt.subplots(figsize=(12, 6))
        plt.bar(daily_tips_sum['time_order'], daily_tips_sum['total_tips'], 
               alpha=0.7, color='orange')
        ax9.set_title('Суммарные чаевые по дням')
        ax9.set_xlabel('Дата')
        ax9.set_ylabel('Сумма чаевых ($)')
        plt.xticks(rotation=45)
        ax9.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig9)
        
        # Кнопка скачивания графика 9
        buf9 = io.BytesIO()
        fig9.savefig(buf9, format='png', dpi=300, bbox_inches='tight')
        st.download_button(
            label="Скачать график динамики чаевых",
            data=buf9.getvalue(),
            file_name="daily_tips_trend.png",
            mime="image/png",
            key="download_trend"
        )

with tab3:
    st.header("Аналитическая информация")
    
    if all(col in filtered_df.columns for col in ['total_bill', 'tip']):
        # Расчет процента чаевых
        filtered_df['tip_percentage'] = (filtered_df['tip'] / filtered_df['total_bill']) * 100
        
        col9, col10, col11 = st.columns(3)
        
        with col9:
            st.metric("Средний % чаевых", f"{filtered_df['tip_percentage'].mean():.1f}%")
        with col10:
            st.metric("Медианный % чаевых", f"{filtered_df['tip_percentage'].median():.1f}%")
        with col11:
            st.metric("Максимальный % чаевых", f"{filtered_df['tip_percentage'].max():.1f}%")
        
        # Матрица корреляций
        st.subheader("Матрица корреляций")
        
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = filtered_df[numeric_cols].corr()
            
            fig10, ax10 = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8},
                ax=ax10
            )
            ax10.set_title('Матрица корреляций численных переменных', fontsize=12)
            st.pyplot(fig10)
            
            # Кнопка скачивания матрицы корреляций
            buf10 = io.BytesIO()
            fig10.savefig(buf10, format='png', dpi=300, bbox_inches='tight')
            st.download_button(
                label="Скачать матрицу корреляций",
                data=buf10.getvalue(),
                file_name="correlation_matrix.png",
                mime="image/png",
                key="download_corr"
            )
        
        # Анализ по категориям
        st.subheader("Анализ по категориям")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            if 'sex' in filtered_df.columns:
                st.write("**Средние чаевые по полу:**")
                sex_tips = filtered_df.groupby('sex', observed=True).agg({
                    'tip': ['mean', 'std'],
                    'total_bill': 'mean',
                    'tip_percentage': 'mean'
                }).round(2)
                st.dataframe(sex_tips, width='stretch')
        
        with analysis_col2:
            if 'time' in filtered_df.columns:
                st.write("**Средние чаевые по времени:**")
                time_tips = filtered_df.groupby('time', observed=True).agg({
                    'tip': ['mean', 'std'],
                    'total_bill': 'mean',
                    'tip_percentage': 'mean'
                }).round(2)
                st.dataframe(time_tips, width='stretch')
        
        with analysis_col3:
            if 'day' in filtered_df.columns:
                st.write("**Средние чаевые по дням:**")
                day_tips = filtered_df.groupby('day', observed=True).agg({
                    'tip': ['mean', 'std'],
                    'total_bill': 'mean',
                    'tip_percentage': 'mean'
                }).round(2)
                st.dataframe(day_tips, width='stretch')

with tab4:
    st.header("Экспорт данных и графиков")
    
    st.subheader("Экспорт данных")
    
    # Экспорт отфильтрованных данных (оригинальные данные)
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="Скачать отфильтрованные данные (CSV)",
        data=csv_data,
        file_name="filtered_tips_data.csv",
        mime="text/csv",
        key="download_csv"
    )
    
    # Экспорт статистики
    stats_data = filtered_df.describe().to_csv()
    st.download_button(
        label="Скачать статистику (CSV)",
        data=stats_data,
        file_name="tips_statistics.csv",
        mime="text/csv",
        key="download_stats"
    )
    
    st.subheader("Создать пользовательский график")
    
    # Пользовательский выбор для графика
    plot_type = st.selectbox(
        "Тип графика",
        ["scatter", "bar", "line", "histogram", "boxplot"]
    )
    
    available_columns = filtered_df.columns.tolist()
    x_axis = st.selectbox("Ось X", available_columns)
    
    if plot_type != "histogram":
        y_axis = st.selectbox("Ось Y", available_columns)
    else:
        y_axis = None
    
    hue_var = st.selectbox("Группировка (hue)", ["Нет"] + available_columns)
    hue_var = None if hue_var == "Нет" else hue_var
    
    if st.button("Создать пользовательский график"):
        fig_custom, ax_custom = plt.subplots(figsize=(10, 6))
        
        try:
            if plot_type == "scatter":
                sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue=hue_var, 
                              ax=ax_custom, palette=palette if hue_var else None)
            elif plot_type == "bar":
                sns.barplot(data=filtered_df, x=x_axis, y=y_axis, hue=hue_var,
                          ax=ax_custom, palette=palette if hue_var else None)
            elif plot_type == "line":
                sns.lineplot(data=filtered_df, x=x_axis, y=y_axis, hue=hue_var,
                           ax=ax_custom, palette=palette if hue_var else None)
            elif plot_type == "histogram":
                sns.histplot(data=filtered_df, x=x_axis, hue=hue_var, 
                           ax=ax_custom, kde=True, palette=palette if hue_var else None)
            elif plot_type == "boxplot":
                sns.boxplot(data=filtered_df, x=x_axis, y=y_axis, hue=hue_var,
                          ax=ax_custom, palette=palette if hue_var else None)
            
            title = f'{plot_type.capitalize()} Plot: {x_axis}' + (f' vs {y_axis}' if y_axis else '')
            ax_custom.set_title(title)
            ax_custom.grid(True, alpha=0.3)
            st.pyplot(fig_custom)
            
            # Кнопка скачивания пользовательского графика
            buf_custom = io.BytesIO()
            fig_custom.savefig(buf_custom, format='png', dpi=300, bbox_inches='tight')
            st.download_button(
                label="Скачать пользовательский график",
                data=buf_custom.getvalue(),
                file_name="custom_plot.png",
                mime="image/png",
                key="download_custom"
            )
            
        except Exception as e:
            st.error(f"Ошибка при создании графика: {e}")

