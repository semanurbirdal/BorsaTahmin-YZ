# 📈 ProFinans AI: Yapay Zeka Tabanlı Borsa Analiz Terminali

**ProFinans AI**, finansal piyasalardaki karmaşık verileri anlamlı içgörülere dönüştüren, makine öğrenmesi destekli bir analiz platformudur. Yahoo Finance verilerini kullanarak hisse senetleri ve kripto paralar üzerinde teknik analiz yapar ve gelecek fiyat eğilimlerini tahmin eder.

🚀 **[Uygulamayı Canlı İzlemek İçin Tıklayın](https://sm-borsa-tahminyz.streamlit.app/)**

---

## 🔥 Temel Özellikler

* **Gerçek Zamanlı Veri Akışı:** `yfinance` API entegrasyonu ile küresel piyasalardan anlık veriler çekilir.
* **Akıllı Tahmin Motoru:** Scikit-learn tabanlı Lineer Regresyon modeli ile 5 günlük fiyat projeksiyonu oluşturulur.
* **İnteraktif Teknik Analiz:** Plotly ile yüksek çözünürlüklü interaktif mum grafikleri sunulur.
* **Modern Arayüz:** Streamlit ile geliştirilen karanlık mod destekli dashboard tasarımı.

---

## ⚙️ Teknik Altyapı
Uygulama, geçmiş fiyat hareketlerini aşağıdaki Lineer Regresyon denklemi üzerinden analiz eder:

$$y = \beta_0 + \beta_1x + \epsilon$$

* **Veri İşleme:** Pandas ve NumPy ile zaman serisi analizi.
* **Tahmin Algoritması:** Geçmiş fiyatların gün sayısına göre eğimi hesaplanarak gelecekteki olası kapanış değerleri tahmin edilir.

---

## 📂 Proje Mimarisi
* `app.py`: UI ve Dashboard yönetimi.
* `tahmin.py`: ML model hesaplamaları.
* `requirements.txt`: Kütüphane bağımlılıkları.

---

## 🛠️ Kurulum
1. `git clone https://github.com/semanurbirdal/BorsaTahmin-YZ.git`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

---
📩 **İletişim:** [sema34birdal@gmail.com]
