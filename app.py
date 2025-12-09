import streamlit as st
import pandas as pd
import joblib
import os
from utils import full_preprocessing, standardize_mileage, name_extract, new_torque_extract
import streamlit.components.v1 as components
import plotly.express as px
import numpy as np
from phik import phik_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image


st.set_page_config(layout="wide") # ширина страницы
background_path = os.path.join(os.path.dirname(__file__), "background.png") # путь к картинке для бэкграунда
with open(background_path, "rb") as bg_file:
    background_base64 = base64.b64encode(bg_file.read()).decode()
background_url = f"url('data:image/png;base64,{background_base64}')"

# стиль
app_style = f"""
<style>
[data-testid="stAppViewContainer"] {{
    margin-left: 0 px;
    background-image: {background_url};
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{
    background-color: rgba(0, 0, 0, 0);
}}

h1, h2, h3 {{
    color: #FFD700;
}}
</style>
"""
st.markdown(app_style, unsafe_allow_html=True)

# Заголовок
st.title("Предсказание цены автомобиля")

#спасибо чату гпт за помощь с красивым сайдбаром

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "EDA"
# функция переключения вкладки
def switch_tab(tab_name):
    st.session_state.active_tab = tab_name
# сайдбар 
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Навигация</h2>", unsafe_allow_html=True)
    if st.button("Посмотреть EDA", use_container_width=True):
        switch_tab("EDA")
    if st.button("Узнать стоимость автомобиля", use_container_width=True):
        switch_tab("Model")
    if st.button("Посмотреть веса модели", use_container_width=True):
        switch_tab("Weights")

# Загрузка датасета 
df_path = os.path.join(os.path.dirname(__file__), "df.csv")
df = pd.read_csv(df_path)

# Загрузка модели
model_path = os.path.join(os.path.dirname(__file__), 'cars_model.pkl')
model = joblib.load(model_path)

# наполнение разделов
if st.session_state.active_tab == "EDA":
    # разделы
    tab1, tab2 = st.tabs(["Визуализация зависимостей", "Тепловая карта"])

    # спасибо чату за красивую кнопочку на репорт
    profile_path = os.path.join(os.path.dirname(__file__), 'processed_train_profile.html')
    with open(profile_path, "rb") as f:
        base64_report = base64.b64encode(f.read()).decode()
    link_html = f"""
    <div style="
        background-color: #eef2fa; 
        color: #21557f; 
        padding: 1rem; 
        border-radius: 8px; 
        font-size: 16px;">
        <a download="EDA_Report.html" 
           href="data:text/html;base64,{base64_report}" 
           style="color: #21557f; text-decoration: none;">
           📄 Скачать полный Profile Report
        </a>
    </div>
    """
    st.markdown(link_html, unsafe_allow_html=True)


    with tab1:
        st.header("Исследование зависимостей между признаками")
        
        selected_features = st.multiselect(
        "Выберите вещественные признаки:",
        df.select_dtypes(include='number').columns.tolist()
    )
        color_by = st.selectbox(
        "Выберите категориальные признаки:",
        df.drop(columns='conf').select_dtypes(include='object').columns.tolist()
    )
        if st.button("Построить график зависимостей"):
            if len(selected_features) >= 2:
                fig = px.scatter_matrix(
                    df,
                    dimensions=selected_features,
                    color=color_by,
                    height=800
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Выберите хотя бы два признака для построения графика.")
    with tab2:
        st.header("Комбинация матриц корреляции Спирмана (числовые фичи) и Phik (категориальные фичи)")
        hm_df = df.drop(columns=['conf'])
        phik_corr = hm_df.phik_matrix()
        colms = hm_df.columns
        corr = np.zeros((len(colms), len(colms)))
        for i in range(len(colms)):
            for j in range(len(colms)):
              col_i = colms[i]
              col_j = colms[j]
              if col_i in df.select_dtypes(include='number').columns  and col_j in df.select_dtypes(include='number').columns:
                corr[i, j] = hm_df[[col_i, col_j]].corr(method='spearman').iloc[0, 1]
              else:
                corr[i, j] = phik_corr.loc[col_i, col_j]
        corr_df = pd.DataFrame(corr, index=colms, columns=colms)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            corr_df,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            annot_kws={"size": 6},
            xticklabels=True,
            yticklabels=True,
            linewidths=0.5,
            cbar_kws={'label': 'Корреляция'}
        )
        ax.tick_params(axis='x', labelsize=6, rotation=45)
        ax.tick_params(axis='y', labelsize=6)
        ax.collections[0].colorbar.ax.tick_params(labelsize=6)
        plt.tight_layout()
        
        # Сохраняем график в буфер
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches='tight') 
        buf.seek(0)
        
        # Вставляем как изображение
        image = Image.open(buf)
        st.image(image, caption="Матрица корреляции", width=800)
        
elif st.session_state.active_tab == "Model":
    st.subheader("Введите данные или загрузите CSV файл для расчета стоимости автомобилей")   
       
    # Форма ввода
    brand = st.selectbox('Марка автомобиля', ['Maruti', 'Skoda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
           'Mahindra', 'Honda', 'Chevrolet', 'Fiat', 'Datsun', 'Tata', 'Jeep',
           'Mercedes', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan',
           'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Force',
           'Ambassador', 'Isuzu', 'Peugeot'  ])
    model_input = st.selectbox('Модель автомобиля', ['Swift', 'Rapid', 'i20', 'Xcent', 'Wagon', '800', 'Etios', 'Figo',
           'Duster', 'Zen', 'KUV', 'Alto', 'Verito', 'WR', 'SX4', 'Baleno',
           'Enjoy', 'Omni', 'Vitara', 'Palio', 'Verna', 'GO', 'Safari',
           'Compass', 'City', 'Fortuner', 'Innova', 'Benz', 'Amaze', 'Pajero',
           'Jazz', 'A6', 'Manza', 'i10', 'Ameo', 'Ertiga', 'Indica', 'Vento',
           'EcoSport', 'X1', 'Celerio', 'Polo', 'Scorpio', 'Freestyle',
           'Passat', 'XUV500', 'Indigo', 'Corolla', 'Terrano', 'Creta',
           'KWID', 'Santro', 'Q5', 'ES', 'XF', 'Rover', '5', 'X4', 'Superb',
           'Hector', 'XC40', 'Q7', 'Ciaz', 'XE', 'Nexon', 'Elantra', 'Glanza',
           '3', 'Camry', 'XC90', 'Ritz', 'Grand', 'Matiz', 'Zest', 'Getz',
           'Tigor', 'Hexa', 'Sunny', 'Ssangyong', 'Quanto', 'Eeco', 'Accent',
           'Ignis', 'Marazzo', 'Tiago', 'Elite', 'Thar', 'Brio', 'Bolero',
           'Beat', 'Willys', 'Micra', 'A', 'Nano', 'GTI', 'V40', 'CR',
           'RediGO', 'Captiva', 'Fiesta', 'Seltos', 'Civic', 'New', 'Sail',
           'Venture', 'Estilo', 'Classic', 'BR', 'EON', 'Aria', 'Sumo', 'TUV',
           'Bolt', 'Accord', 'Grande', 'S', 'Yaris', 'Xylo', 'Tavera',
           'Linea', 'Endeavour', 'Aveo', 'Esteem', 'Triber', 'Fusion',
           'Octavia', 'A4', 'XL6', 'Santa', 'Spark', 'Ecosport', 'Punto',
           'Optra', 'Mobilio', 'Qualis', 'BRV', 'X6', 'Cruze', '6', 'Jeep',
           'Lodgy', 'Pulse', 'Supro', 'Ingenio', 'Renault', 'Wrangler',
           'Kicks', 'NuvoSport', 'Jetta', 'Aspire', 'Teana', 'Yeti', 'Q3',
           'Gurkha', 'Logan', 'A3', 'XUV300', 'Dzire', 'Ikon', 'Fluence',
           'Xenon', 'One', '7', 'S60', 'Lancer', 'X7', 'Premio', 'Fabia',
           'Platinum', 'Captur', 'Gypsy', 'Estate', 'Koleos', 'CLASSIC',
           'Harrier', 'Multivan', 'Avventura', 'Laura', 'Sonata', 'MUX',
           'Tucson', 'Winger', 'Spacio', 'CrossPolo', 'Marshal', 'D', 'X3',
           'Land', '309', 'Trailblazer', 'MU', 'Venue', 'Scala', 'S90'])
    conf = st.text_input("Конфигурация (например: 1 5 TDI Ambition)")
    year = st.number_input("Год выпуска", min_value=1990, max_value=2025, value=2015)
    engine = st.text_input("Объем двигателя (например: 1248 CC)")
    max_power = st.text_input("Мощность (например: 74 bhp)")
    mileage = st.text_input("Расход топлива (например: 23.4 kmpl)")
    torque = st.text_input("Крутящий момент (например: 190Nm@ 2000rpm)")
    seats = st.number_input("Количество мест", min_value=2, max_value=14, value=5)
    fuel = st.selectbox("Тип топлива", ["Petrol", "Diesel", "LPG", "CNG"])
    transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
    seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
    owner = st.selectbox("Количество владельцев", ['First Owner', 'Second Owner', 'Third Owner',
           'Fourth & Above Owner', 'Test Drive Car'])
    km_driven = st.number_input("Пробег (в км)", min_value=0)
    
    # ввод через CSV
    st.header("Предсказание из CSV-файла")
    uploaded_file = st.file_uploader("Загрузите датасет", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        predictions = model.predict(input_df)
        input_df['predicted_price'] = predictions.astype(int)
        st.subheader("Результаты предсказания:")
        st.write(input_df)
        
    # запуск расчетов 
    if st.button("Узнать стоимость"):
        # формируем датафрейм
        input_df = pd.DataFrame([{
            'name': f"{brand} {model_input} {conf}",
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'torque': torque,
            'seats': seats
        }])
    
        # предсказание
        predicted_price = model.predict(input_df)[0]
        st.success(f"Предполагаемая цена: {predicted_price:,.0f} ₽")

elif st.session_state.active_tab == "Weights":
    # список признаков
    features = model.named_steps['preprocessing'].get_feature_names_out()
    # веса
    coefficients = model.named_steps['model'].regressor_.coef_
    # отображение
    weights = pd.Series(coefficients, index=features)
    st.subheader("Веса признаков обученной модели")
    st.bar_chart(weights)
