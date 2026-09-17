import pickle
import pandas as pd
pydeck_available = True
try:
    import pydeck as pdk
except ImportError:
    pydeck_available = False
import streamlit as st

# Modeli ve scaler'ı yükleme (pickle yöntemiyle)
model = pickle.load(open("CaliforniaHousing_xgb_model.pkl", "rb"))
scaler = pickle.load(open("CaliforniaHousing_xgb_scaler.pkl", "rb"))

st.title("California Ev Fiyatı Tahmin Uygulaması")
st.write(
    "Harita üzerinden konumu seçin veya ayarlayın; yapay zeka modelimiz"
    " özellikleri birleştirerek tahmini ev fiyatını versin!"
)

# --- 1. HARİTA İLE KONUM (ENLEM & BOYLAM) SEÇİMİ ---
st.subheader("📍 Konum Seçimi (California)")

# Varsayılan başlangıç konumu (San Francisco merkezi)
default_lat = 37.7749
default_lon = -122.4194

# Kullanıcının harita üzerinde konum seçebilmesi için DataFrame oluşturuyoruz
map_data = pd.DataFrame({"lat": [default_lat], "lon": [default_lon]})

# İnteraktif harita gösterimi (Kullanıcı haritada gezinebilir)
edited_map_data = st.map(map_data, zoom=6, use_container_width=True)

# Pratik olması adına kullanıcıdan enlem ve boylamı hassas ayarlayabilmesi için
# sayısal girişler sunuyoruz ama bunları harita koordinatlarıyla senkronize tutuyoruz.
col1, col2 = st.columns(2)
with col1:
  latitude = st.number_input(
      "Enlem (Latitude)", value=default_lat, format="%.4f"
  )
with col2:
  longitude = st.number_input(
      "Boylam (Longitude)", value=default_lon, format="%.4f"
  )

# --- 2. DİĞER EV ÖZELLİKLERİ ---
st.subheader("🏠 Ev Özellikleri")

col3, col4 = st.columns(2)
with col3:
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

with col4:
  # Kullanıcıdan "Medyan Gelir" istemek yerine, seçilen bölgeye/varsayılan duruma göre standart atıyoruz
  med_inc = st.number_input(
      "Bölgesel Medyan Gelir (MedInc - On Bin $)",
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

# --- 3. VERİYİ MODELİN BEKLEDİĞİ SIRAYLA BİRLEŞTİRME ---
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

# --- 4. ÖLÇEKLENDİRME VE TAHMİN ---
input_scaled = scaler.transform(input_data)

if st.button("Fiyatı Tahmin Et", type="primary"):
  tahmin = model.predict(input_scaled)
  # California Housing veri seti hedefi 100,000$ cinsindendir
  fiyat_usd = tahmin[0] * 100000
  st.success(f"💰 Tahmini Ev Fiyatı: ${fiyat_usd:,.2f}")
