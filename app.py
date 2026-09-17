import pickle
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Modeli ve scaler'ı yükleme
model = pickle.load(open("CaliforniaHousing_xgb_model.pkl", "rb"))
scaler = pickle.load(open("CaliforniaHousing_xgb_scaler.pkl", "rb"))

st.title("California Ev Fiyatı Tahmin Uygulaması")
st.write(
    "Harita üzerinde evinizin bulunduğu konuma **tıklayın**, yapay zeka"
    " diğer özelliklerle birlikte fiyatı tahmin etsin!"
)

# --- 1. İNTERAKTİF HARİTA İLE KONUM SEÇİMİ ---
st.subheader("📍 Konum Seçimi")

# Harita merkezini California (San Francisco civarı) olarak başlatıyoruz
m = folium.Map(location=[37.7749, -122.4194], zoom_start=6)

# Kullanıcının haritaya tıklayabilmesi için haritayı Streamlit'e aktarıyoruz
map_data = st_folium(m, width=700, height=400)

# Kullanıcı haritaya tıkladıysa koordinatları al, tıklamadıysa varsayılan değer kullan
if map_data and map_data.get("last_clicked"):
  latitude = map_data["last_clicked"]["lat"]
  longitude = map_data["last_clicked"]["lng"]
  st.success(
      f"Seçilen Konum -> Enlem: {latitude:.4f}, Boylam: {longitude:.4f}"
  )
else:
  # Varsayılan başlangıç değerleri
  latitude = 37.7749
  longitude = -122.4194
  st.info(
      "💡 Harita üzerinde henüz bir yere tıklamadınız. Varsayılan olarak"
      " San Francisco merkezi seçilmiştir. Haritaya tıklayarak konumu"
      " değiştirebilirsiniz."
  )

# --- 2. DİĞER EV ÖZELLİKLERİ ---
st.subheader("🏠 Diğer Ev Özellikleri")

col1, col2 = st.columns(2)
with col1:
  house_age = st.number_input(
      "Bina Yaşı (HouseAge)", min_value=1.0, max_value=52.0, value=20.0
  )
  ave_rooms = st.number_input(
      "Ortalama Oda Sayısı (AveRooms)", min_value=1.0, max_value=50.0, value=5.0
  )
  ave_bedrms = st.number_input(
      "Ortalama Yatak Odası (AveBedrms)",
      min_value=0.5,
      max_value=10.0,
      value=1.0,
  )

with col2:
  med_inc = st.number_input(
      "Bölgesel Medyan Gelir (MedInc)",
      min_value=0.5,
      max_value=15.0,
      value=3.5,
  )
  population = st.number_input(
      "Bölge Nüfusu (Population)", min_value=10.0, max_value=35000.0, value=1200.0
  )
  ave_occup = st.number_input(
      "Hane Başına Ortalama Oturan (AveOccup)",
      min_value=1.0,
      max_value=20.0,
      value=3.0,
  )

# --- 3. VERİYİ BİRLEŞTİRME VE TAHMİN ---
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

input_scaled = scaler.transform(input_data)

if st.button("🚀 Fiyatı Tahmin Et", type="primary"):
  tahmin = model.predict(input_scaled)
  fiyat_usd = tahmin[0] * 100000
  st.success(f"💰 Tahmini Ev Fiyatı: ${fiyat_usd:,.2f}")
