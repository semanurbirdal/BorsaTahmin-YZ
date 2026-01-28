import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# --- 1. SAYFA VE TASARIM AYARLARI ---
st.set_page_config(page_title="ProFinans AI Terminal", layout="wide", page_icon="📈")

# Özel CSS: Koyu tema bilgi kartları ve düzenlemeler
st.markdown("""
    <style>
    /* Ana arka planı ve metin renklerini ayarlayalım */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Metrik kutularını özelleştirelim (Kart Görünümü) */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00d1ff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #9ca3af;
    }
    /* Konteynerlara arka plan verelim */
    .css-1r6slb0 {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #31333F;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VARLIK LİSTESİ ---
hisseler = {
    "BIST: THY (THYAO)": "THYAO.IS",
    "BIST: Aselsan (ASELS)": "ASELS.IS",
    "BIST: Ereğli (EREGL)": "EREGL.IS",
    "BIST: Koç Holding (KCHOL)": "KCHOL.IS",
    "ABD: Apple (AAPL)": "AAPL",
    "ABD: Nvidia (NVDA)": "NVDA",
    "ABD: Tesla (TSLA)": "TSLA",
    "Kripto: Bitcoin (BTC)": "BTC-USD",
    "Kripto: Ethereum (ETH)": "ETH-USD",
    "Emtia: Altın Ons": "GC=F"
}

# --- 3. SOL PANEL (KONTROL) ---
with st.sidebar:
    st.title("🎛️ Kontrol Paneli")
    secilen_ad = st.selectbox("Varlık Seçimi", list(hisseler.keys()), index=0)
    hisse_kodu = hisseler[secilen_ad]
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        donem = st.selectbox("Veri Geçmişi", ["6mo", "1y", "2y", "5y"], index=1)
    with col_s2:
        tahmin_gun = st.number_input("Tahmin (Gün)", min_value=7, max_value=90, value=30)
        
    st.markdown("---")
    st.info("Bu panel Lineer Regresyon ve 50 Günlük Hareketli Ortalama (MA50) kullanır.")

# --- 4. VERİ İŞLEME ---
data = yf.download(hisse_kodu, period=donem)

if data.empty or len(data) < 50:
    st.error("Yeterli veri çekilemedi. Lütfen başka bir hisse veya dönem seçin.")
    st.stop()

# Veri Temizliği
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data[['Close', 'Open', 'High', 'Low', 'Volume']].dropna()

# Teknik İndikatör Ekleme (MA50)
data['MA50'] = data['Close'].rolling(window=50).mean()

# --- 5. ANA EKRAN: ÜST BİLGİ KARTLARI ---
st.subheader(f"📊 {secilen_ad} Piyasa Özeti")

# Güncel verileri hesapla
son_fiyat = data['Close'].iloc[-1]
onceki_fiyat = data['Close'].iloc[-2]
degisim = son_fiyat - onceki_fiyat
yuzde_degisim = (degisim / onceki_fiyat) * 100
en_yuksek = data['Close'].max()
en_dusuk = data['Close'].min()

# Yapay Zeka Başarı Skoru Hesapla
X = np.arange(len(data)).reshape(-1, 1)
y = data['Close'].values
model_test = LinearRegression().fit(X, y)
tahmin_test = model_test.predict(X)
hata_mae = mean_absolute_error(y, tahmin_test)
basari_skoru = max(0, 100 - (hata_mae / y.mean() * 100))

# Kartları Yan Yana Diz
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Son Fiyat", f"{son_fiyat:.2f}", f"{yuzde_degisim:+.2f}%", delta_color="inverse")
k2.metric("Günlük Değişim", f"{degisim:+.2f}")
k3.metric("Dönem En Yüksek", f"{en_yuksek:.2f}")
k4.metric("Dönem En Düşük", f"{en_dusuk:.2f}")
k5.metric("YZ Güven Skoru", f"%{basari_skoru:.1f}", delta="Yüksek" if basari_skoru > 90 else "Orta")

st.markdown("---")

# --- 6. ANA EKRAN: GRAFİK VE VERİ TABLOSU ---
col_grafik, col_veri = st.columns([3, 1]) # Ekranı 3'e 1 oranında böl

with col_grafik:
    # --- GELİŞMİŞ GRAFİK ---
    st.subheader("📈 Teknik ve Yapay Zeka Analizi")
    
    # Gelecek Tahmini Hesapla
    model_final = LinearRegression().fit(X, y)
    gelecek_X = np.arange(len(data), len(data) + tahmin_gun).reshape(-1, 1)
    gelecek_tahmin = model_final.predict(gelecek_X)
    gelecek_tarihler = pd.date_range(data.index[-1] + timedelta(days=1), periods=tahmin_gun)

    fig = go.Figure()
    
    # Gerçek Fiyat (Alan Grafiği)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Fiyat", 
                             fill='tozeroy', fillcolor='rgba(0, 209, 255, 0.1)',
                             line=dict(color='#00d1ff', width=2)))
    
    # 50 Günlük Ortalama
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="MA50 (Ortalama)",
                             line=dict(color='#ffbf00', width=1.5)))

    # YZ Trend Tahmini
    fig.add_trace(go.Scatter(x=gelecek_tarihler, y=gelecek_tahmin, name=f"YZ {tahmin_gun} Günlük Trend",
                             line=dict(color='#ff0055', width=3, dash='dot')))

    # Grafik Ayarları (Koyu Tema)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', # Şeffaf arka plan
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        hovermode="x unified",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#31333F'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_veri:
    # --- YAN TABLO ---
    st.subheader("🗓️ Son Veriler")
    son_veriler = data[['Close', 'Volume']].sort_index(ascending=False).head(10)
    st.dataframe(son_veriler, use_container_width=True, height=450)

# --- 7. ALT BİLGİ ---
trend_yonu = "YUKARI ↗️" if model_final.coef_[0] > 0 else "AŞAĞI ↘️"
st.success(f"💡 **YZ Analiz Özeti:** Model, {secilen_ad} için genel trendin **{trend_yonu}** olduğunu öngörüyor. {tahmin_gun} gün sonrası için hedef bölge yaklaşık **{gelecek_tahmin[-1]:.2f}** seviyesidir.")