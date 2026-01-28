import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. VERİ ÇEKME: Türk Hava Yolları (THYAO) verilerini alalım
hisse = yf.download("THYAO.IS", period="1y") 

# 2. VERİ HAZIRLAMA: Sadece 'Close' (Kapanış) fiyatını kullanalım
hisse = hisse[['Close']].dropna()
hisse['Gun'] = np.arange(len(hisse)) # Günleri 0, 1, 2... diye numaralandıralım

# 3. MODELİ EĞİTME: Yapay zekaya geçmiş veriyi gösterelim
X = hisse[['Gun']] # Girdi (Gün sayısı)
y = hisse['Close'] # Çıktı (Fiyat)

model = LinearRegression()
model.fit(X, y) # Model burada öğreniyor

# 4. TAHMİN: Gelecek 30 günü tahmin edelim
gelecek_gunler = np.array([[len(hisse) + i] for i in range(30)])
tahminler = model.predict(gelecek_gunler)

# 5. GÖRSELLEŞTİRME
plt.figure(figsize=(12,6))
plt.plot(hisse['Gun'], hisse['Close'], label="Gerçek Fiyatlar")
plt.plot(range(len(hisse), len(hisse)+30), tahminler, label="30 Günlük Tahmin", linestyle="--")
plt.legend()
plt.title("THYAO Fiyat Tahmini (Basit Regresyon)")
plt.show()