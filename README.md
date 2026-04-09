# 🌐 Web Traffic Time Series Forecasting

Bu proje, saatlik web trafiği verilerini kullanarak oturum sayısını tahmin eden bir **zaman serisi analizi** çalışmasıdır. ARIMA modeli ile gerçekleştirilen tahminler, veri ön işleme, keşifsel veri analizi (EDA) ve model değerlendirme adımlarını kapsamaktadır.

---

## 📁 Proje Yapısı

```
📦 Web-Traffic-Time-Series-Forecasting
├── 📓 WebTrafficTimeSeriesForecasting.ipynb  # Ana analiz notebook'u
├── 📊 webtraffic.csv                          # Ham veri seti
├── 🤖 arima.pkl                               # Eğitilmiş ARIMA modeli (Google Drive)
└── 📄 README.md
```

---

## 📊 Veri Seti

| Özellik | Detay |
|---|---|
| **Dosya** | `webtraffic.csv` |
| **Satır Sayısı** | 4.896 |
| **Sütunlar** | `Hour Index`, `Sessions` |
| **Min Oturum** | 570.856.572 |
| **Max Oturum** | 6.061.858.000 |
| **Ortalama** | ~2.25 Milyar |

---

## 🔬 Yapılan Analizler

- ✅ Keşifsel Veri Analizi (EDA)
- ✅ Zaman serisi görselleştirmesi
- ✅ Oturum dağılımı analizi
- ✅ MinMax normalizasyonu
- ✅ %80 / %20 train-test bölümü
- ✅ ADF Durağanlık Testi
- ✅ ACF / PACF grafikleri
- ✅ ARIMA model eğitimi ve tahmini

---

## 🤖 Model

**ARIMA** (AutoRegressive Integrated Moving Average) modeli kullanılmıştır.

> ⚠️ Eğitilmiş model dosyası (`arima.pkl`) boyutu nedeniyle GitHub'a yüklenememiştir.  
> 📥 **[arima.pkl dosyasını buradan indirebilirsiniz](https://drive.google.com/file/d/1kLvcI90xYwM40KSvpXbz0WGyQlZAHpG9/view?usp=drive_link)**

---

## 🚀 Kurulum ve Kullanım

### Gereksinimler

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### Çalıştırma

```bash
git clone https://github.com/tugcesi/Web-Traffic-Time-Series-Forecasting.git
cd Web-Traffic-Time-Series-Forecasting
jupyter notebook WebTrafficTimeSeriesForecasting.ipynb
```

---

## 🛠️ Kullanılan Teknolojiler

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-library-green?logo=pandas)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 📄 Lisans

Bu proje [MIT License](LICENSE) ile lisanslanmıştır.