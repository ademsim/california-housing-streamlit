import pickle
import pandas as pd
import streamlit as st

# Modeli ve scaler'ı yükleme 
model = pickle.load(open("CaliforniaHousing_xgb_model.pkl", "rb"))
scaler = pickle.load(open("CaliforniaHousing_xgb_scaler.pkl", "rb"))

st.title("California Ev Fiyatı Tahmin Uygulaması")
st.write(
    "Lütfen ev ve konum özelliklerini girerek tahmini fiyatı hesaplayın."
)

# 1. Tüm sütunlar için Streamlit input bileşenleri
med_inc = st.number_input(
    "Medyan Gelir (MedInc)", value=3.87, format="%.4f"
)
house_age = st.number_input(
    "Bina Yaşı (HouseAge)", value=28.0, format="%.1f"
)
ave_rooms = st.number_input(
    "Ortalama Oda Sayısı (AveRooms)", value=5.43, format="%.4f"
)
ave_bedrms = st.number_input(
    "Ortalama Yatak Odası (AveBedrms)", value=1.10, format="%.4f"
)
population = st.number_input(
    "Bölge Nüfusu (Population)", value=1425.0, format="%.1f"
)
ave_occup = st.number_input(
    "Hane Başına Ortalama Oturan (AveOccup)", value=3.07, format="%.4f"
)
latitude = st.number_input("Enlem (Latitude)", value=37.88, format="%.4f")
longitude = st.number_input("Boylam (Longitude)", value=-122.23, format="%.4f")

# 2. Girdileri modelin eğitildiği sırayla DataFrame'e aktarma
input_data = pd.DataFrame(
    [[
        med_inc,
        house_age,
        ave_rooms,
        ave_bedrms,
        population,
        ave_occup,
        latitude,
        longitude,
    ]],
    columns=[
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ],
)

# 3. Ölçeklendirme
input_scaled = scaler.transform(input_data)

# 4. Tahmin Yapma
if st.button("Fiyatı Tahmin Et"):
  tahmin = model.predict(input_scaled)
  # California Housing hedef değeri 100,000 USD cinsinden olduğu için çarpıyoruz
  st.success(f"Tahmini Ev Fiyatı: ${tahmin[0] * 100000:,.2f}")
