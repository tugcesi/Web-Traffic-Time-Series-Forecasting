import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import pickle
import io

st.set_page_config(
    page_title="Web Trafiği Tahmini",
    page_icon="🌐",
    layout="wide"
)

PRIMARY_COLOR = "#FF6B6B"
SECONDARY_COLOR = "#20B2AA"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("webtraffic.csv")
    except FileNotFoundError:
        st.error(
            "❌ `webtraffic.csv` dosyası bulunamadı. "
            "Lütfen dosyanın uygulama ile aynı dizinde olduğundan emin olun."
        )
        st.stop()
    return df


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🌐 Web Trafiği Tahmini")
st.sidebar.markdown("Saatlik web trafiği verilerini analiz eden ve ARIMA modeliyle tahmin yapan interaktif uygulama.")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Sayfa Seçin",
    ["🏠 Ana Sayfa", "📊 Keşifsel Veri Analizi", "🤖 Tahmin", "📈 Model Bilgisi"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📂 Proje Linkleri**")
st.sidebar.markdown("[GitHub Repo](https://github.com/tugcesi/Web-Traffic-Time-Series-Forecasting)")
st.sidebar.markdown("[ARIMA Model (Google Drive)](https://drive.google.com/file/d/1kLvcI90xYwM40KSvpXbz0WGyQlZAHpG9/view?usp=drive_link)")

# ===========================================================================
# 🏠 SAYFA 1: ANA SAYFA
# ===========================================================================

if page == "🏠 Ana Sayfa":
    st.title("🌐 Web Trafiği Zaman Serisi Tahmini")
    st.markdown(
        """
        Bu uygulama, saatlik web trafiği verilerini **ARIMA** (AutoRegressive Integrated Moving Average)
        modeli kullanarak analiz eder ve gelecekteki oturum sayılarını tahmin eder.

        - 📊 **4.896 saatlik** gözlem verisi
        - 🤖 **ARIMA(5,1,2)** modeli ile tahmin
        - 📈 **MinMaxScaler** ile normalize edilmiş eğitim süreci
        """
    )
    st.markdown("---")

    # Veri önizleme
    st.subheader("📋 Veri Önizleme (İlk 10 Satır)")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # Temel istatistikler
    st.subheader("📊 Temel İstatistikler")
    sessions = df["Sessions"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📉 Minimum", f"{sessions.min():,.0f}")
    col2.metric("📈 Maksimum", f"{sessions.max():,.0f}")
    col3.metric("📊 Ortalama", f"{sessions.mean():,.0f}")
    col4.metric("📐 Standart Sapma", f"{sessions.std():,.0f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.info(f"**Toplam Gözlem Sayısı:** {len(df):,} saat")
    with col_right:
        st.info(
            f"**Veri Aralığı:** Hour Index {int(df['Hour Index'].min())} — {int(df['Hour Index'].max())}"
        )

# ===========================================================================
# 📊 SAYFA 2: KEŞİFSEL VERİ ANALİZİ (EDA)
# ===========================================================================

elif page == "📊 Keşifsel Veri Analizi":
    st.title("📊 Keşifsel Veri Analizi")
    st.markdown("Veri setinin detaylı görsel analizi.")
    st.markdown("---")

    # Zaman serisi grafiği
    st.subheader("📈 Zaman Serisi — Saatlik Oturum Sayısı")
    fig_ts = px.line(
        df,
        x="Hour Index",
        y="Sessions",
        title="Saatlik Web Trafiği (Hour Index vs Sessions)",
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig_ts.update_layout(xaxis_title="Saat İndeksi", yaxis_title="Oturum Sayısı")
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # Histogram
    st.subheader("📊 Oturum Sayısı Dağılımı")
    fig_hist = px.histogram(
        df,
        x="Sessions",
        nbins=50,
        title="Oturum Sayısı Histogram Dağılımı",
        color_discrete_sequence=[SECONDARY_COLOR],
    )
    fig_hist.update_layout(xaxis_title="Oturum Sayısı", yaxis_title="Frekans")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Rolling mean
    st.subheader("📉 Hareketli Ortalama (24 Saatlik Pencere)")
    df_roll = df.copy()
    df_roll["Rolling Mean (24h)"] = df_roll["Sessions"].rolling(window=24).mean()

    fig_roll = go.Figure()
    fig_roll.add_trace(
        go.Scatter(
            x=df_roll["Hour Index"],
            y=df_roll["Sessions"],
            name="Gerçek Değer",
            line=dict(color=PRIMARY_COLOR, width=1),
            opacity=0.5,
        )
    )
    fig_roll.add_trace(
        go.Scatter(
            x=df_roll["Hour Index"],
            y=df_roll["Rolling Mean (24h)"],
            name="Hareketli Ortalama (24h)",
            line=dict(color=SECONDARY_COLOR, width=2),
        )
    )
    fig_roll.update_layout(
        title="Gerçek Değer vs 24 Saatlik Hareketli Ortalama",
        xaxis_title="Saat İndeksi",
        yaxis_title="Oturum Sayısı",
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    st.markdown("---")

    # ADF testi
    st.subheader("🧪 ADF Durağanlık Testi Sonucu")
    with st.spinner("ADF testi hesaplanıyor..."):
        adf_result = adfuller(df["Sessions"].dropna())

    adf_col1, adf_col2, adf_col3 = st.columns(3)
    adf_col1.metric("ADF Test İstatistiği", f"{adf_result[0]:.4f}")
    adf_col2.metric("p-değeri", f"{adf_result[1]:.6f}")
    adf_col3.metric("Kullanılan Gecikme Sayısı", str(adf_result[2]))

    if adf_result[1] < 0.05:
        st.success("✅ Seri **durağandır** (p < 0.05). H₀ (birim kök var) reddedildi.")
    else:
        st.warning("⚠️ Seri **durağan değildir** (p ≥ 0.05). Fark alma gerekebilir.")

    st.markdown("**Kritik Değerler:**")
    crit_data = {
        "Güven Seviyesi": ["% 1", "% 5", "% 10"],
        "Kritik Değer": [
            f"{adf_result[4]['1%']:.4f}",
            f"{adf_result[4]['5%']:.4f}",
            f"{adf_result[4]['10%']:.4f}",
        ],
    }
    st.dataframe(pd.DataFrame(crit_data), use_container_width=False)

# ===========================================================================
# 🤖 SAYFA 3: TAHMİN
# ===========================================================================

elif page == "🤖 Tahmin":
    st.title("🤖 Tahmin")
    st.markdown("ARIMA modeli ile gelecekteki web trafiğini tahmin edin.")
    st.markdown("---")

    # Model yükleme
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Model Yükle")
    uploaded_file = st.sidebar.file_uploader(
        "arima.pkl dosyasını yükleyin",
        type=["pkl"],
        help="Eğitilmiş ARIMA modelini pickle formatında yükleyin.",
    )

    model = None
    if uploaded_file is not None:
        try:
            model = pickle.load(uploaded_file)
            st.sidebar.success("✅ Model başarıyla yüklendi!")
        except Exception as e:
            st.sidebar.error(f"❌ Model yüklenemedi: {e}")

    # ---------------------------------------------------------------------------
    # Model yüklendiyse: ARIMA tahmini
    # ---------------------------------------------------------------------------
    if model is not None:
        st.subheader("🔮 ARIMA Tahmin Sonuçları")

        steps = st.slider(
            "Kaç saat ilerisi tahmin edilsin?",
            min_value=1,
            max_value=168,
            value=24,
            step=1,
        )

        apply_inverse = st.checkbox(
            "Inverse transform uygula (model normalize edilmiş veri üzerinde eğitildiyse işaretleyin)",
            value=False,
        )

        if st.button("Tahmin Et", type="primary"):
            with st.spinner("Tahmin yapılıyor..."):
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(df[["Sessions"]])

                try:
                    forecast_result = model.forecast(steps=steps)
                    forecast_values = np.array(forecast_result).reshape(-1, 1)

                    if apply_inverse:
                        predicted_sessions = scaler.inverse_transform(forecast_values).flatten()
                    else:
                        predicted_sessions = forecast_values.flatten()

                except Exception as e:
                    st.error(f"Tahmin sırasında hata oluştu: {e}")
                    predicted_sessions = None

            if predicted_sessions is not None:
                last_hour = int(df["Hour Index"].max())
                future_hours = np.arange(last_hour + 1, last_hour + 1 + steps)

                forecast_df = pd.DataFrame(
                    {"Hour Index": future_hours, "Predicted Sessions": predicted_sessions.astype(int)}
                )

                # Grafik: Gerçek + Tahmin
                fig_pred = go.Figure()
                fig_pred.add_trace(
                    go.Scatter(
                        x=df["Hour Index"].tail(200),
                        y=df["Sessions"].tail(200),
                        name="Gerçek Değer",
                        line=dict(color=PRIMARY_COLOR),
                    )
                )
                fig_pred.add_trace(
                    go.Scatter(
                        x=forecast_df["Hour Index"],
                        y=forecast_df["Predicted Sessions"],
                        name="ARIMA Tahmini",
                        line=dict(color=SECONDARY_COLOR, dash="dash"),
                    )
                )
                fig_pred.update_layout(
                    title=f"Gerçek Web Trafiği + {steps} Saatlik ARIMA Tahmini",
                    xaxis_title="Saat İndeksi",
                    yaxis_title="Oturum Sayısı",
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # Tahmin tablosu
                st.subheader("📋 Tahmin Sonuçları Tablosu")
                st.dataframe(forecast_df, use_container_width=True)

                # CSV indirme
                csv_buffer = io.StringIO()
                forecast_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Tahminleri CSV Olarak İndir",
                    data=csv_buffer.getvalue(),
                    file_name="arima_tahminleri.csv",
                    mime="text/csv",
                )

    # ---------------------------------------------------------------------------
    # Model yüklenmediyse: Uyarı + Hareketli Ortalama alternatifi
    # ---------------------------------------------------------------------------
    else:
        st.warning(
            "⚠️ ARIMA modeli henüz yüklenmedi. Sol panelden `arima.pkl` dosyasını yükleyin."
        )
        st.markdown(
            "📥 **[arima.pkl dosyasını Google Drive'dan indirebilirsiniz]"
            "(https://drive.google.com/file/d/1kLvcI90xYwM40KSvpXbz0WGyQlZAHpG9/view?usp=drive_link)**"
        )
        st.markdown("---")

        st.subheader("📉 Alternatif: Hareketli Ortalama Tahmini")
        st.info(
            "Model yüklenmediği için basit hareketli ortalama (Moving Average) tahmini gösterilmektedir."
        )

        ma_window = st.slider(
            "Hareketli Ortalama Pencere Boyutu (saat)",
            min_value=6,
            max_value=168,
            value=24,
            step=6,
        )
        ma_steps = st.slider(
            "Kaç saat ilerisi tahmin edilsin?",
            min_value=1,
            max_value=168,
            value=24,
            step=1,
        )

        if st.button("Hareketli Ortalama ile Tahmin Et"):
            last_window = df["Sessions"].tail(ma_window).values
            ma_value = last_window.mean()
            last_hour = int(df["Hour Index"].max())
            future_hours = np.arange(last_hour + 1, last_hour + 1 + ma_steps)
            ma_forecast = np.full(ma_steps, ma_value)

            ma_df = pd.DataFrame(
                {"Hour Index": future_hours, "MA Tahmini": ma_forecast.astype(int)}
            )

            fig_ma = go.Figure()
            fig_ma.add_trace(
                go.Scatter(
                    x=df["Hour Index"].tail(200),
                    y=df["Sessions"].tail(200),
                    name="Gerçek Değer",
                    line=dict(color=PRIMARY_COLOR),
                )
            )
            fig_ma.add_trace(
                go.Scatter(
                    x=ma_df["Hour Index"],
                    y=ma_df["MA Tahmini"],
                    name=f"Hareketli Ortalama Tahmini ({ma_window}h pencere)",
                    line=dict(color=SECONDARY_COLOR, dash="dash"),
                )
            )
            fig_ma.update_layout(
                title=f"Gerçek Web Trafiği + {ma_steps} Saatlik Hareketli Ortalama Tahmini",
                xaxis_title="Saat İndeksi",
                yaxis_title="Oturum Sayısı",
            )
            st.plotly_chart(fig_ma, use_container_width=True)

            st.dataframe(ma_df, use_container_width=True)

            csv_buffer = io.StringIO()
            ma_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Tahminleri CSV Olarak İndir",
                data=csv_buffer.getvalue(),
                file_name="ma_tahminleri.csv",
                mime="text/csv",
            )

# ===========================================================================
# 📈 SAYFA 4: MODEL BİLGİSİ
# ===========================================================================

elif page == "📈 Model Bilgisi":
    st.title("📈 Model Bilgisi")
    st.markdown("---")

    st.subheader("🤖 ARIMA Nedir?")
    st.markdown(
        """
        **ARIMA** (AutoRegressive Integrated Moving Average), zaman serisi tahmininde yaygın
        olarak kullanılan istatistiksel bir modeldir.

        - **AR (AutoRegressive — Oto-regresif):** Geçmiş gözlemlerin ağırlıklı toplamını kullanır.
        - **I (Integrated — Entegre):** Zaman serisini durağan hale getirmek için fark alır.
        - **MA (Moving Average — Hareketli Ortalama):** Geçmiş tahmin hatalarının ağırlıklı toplamını kullanır.
        """
    )

    st.markdown("---")

    st.subheader("⚙️ Model Parametreleri")
    params_df = pd.DataFrame(
        {
            "Parametre": ["p (AR terimi)", "d (Fark derecesi)", "q (MA terimi)", "Model adı"],
            "Değer": ["5", "1", "2", "ARIMA(5,1,2)"],
            "Açıklama": [
                "5 gecikmeli oto-regresif terim",
                "1. dereceden fark alınarak durağanlaştırma",
                "2 gecikmeli hareketli ortalama terimi",
                "Toplam model yapılandırması",
            ],
        }
    )
    st.dataframe(params_df, use_container_width=True)

    st.markdown("---")

    st.subheader("📦 Kullanılan Kütüphaneler")
    libs_df = pd.DataFrame(
        {
            "Kütüphane": [
                "streamlit",
                "pandas",
                "numpy",
                "plotly",
                "scikit-learn",
                "statsmodels",
                "matplotlib",
                "seaborn",
            ],
            "Versiyon (min)": [
                "≥ 1.32.0",
                "≥ 2.0.0",
                "≥ 1.24.0",
                "≥ 5.15.0",
                "≥ 1.3.0",
                "≥ 0.14.0",
                "≥ 3.7.0",
                "≥ 0.12.0",
            ],
            "Kullanım Amacı": [
                "Web uygulaması çerçevesi",
                "Veri işleme",
                "Sayısal hesaplama",
                "İnteraktif görselleştirme",
                "MinMaxScaler normalizasyonu",
                "ARIMA modeli ve ADF testi",
                "Ek görselleştirme",
                "İstatistiksel görselleştirme",
            ],
        }
    )
    st.dataframe(libs_df, use_container_width=True)

    st.markdown("---")

    st.subheader("🔢 Normalizasyon Bilgisi")
    st.markdown(
        """
        Eğitim sürecinde veriler **MinMaxScaler** ile normalize edilmiştir:

        - **Yöntem:** `sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))`
        - **Aralık:** 0 ile 1 arasında
        - **Eğitim seti:** İlk **%80** (~3.917 satır)
        - **Test seti:** Son **%20** (~979 satır)
        - **Tahmin sonrası:** `inverse_transform` ile gerçek değer aralığına geri çevrilir
        """
    )

    st.markdown("---")

    st.subheader("🔗 Proje Linkleri")
    col1, col2 = st.columns(2)
    col1.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)]"
        "(https://github.com/tugcesi/Web-Traffic-Time-Series-Forecasting)"
    )
    col2.markdown(
        "[![Google Drive](https://img.shields.io/badge/Google%20Drive-arima.pkl-blue?logo=googledrive)]"
        "(https://drive.google.com/file/d/1kLvcI90xYwM40KSvpXbz0WGyQlZAHpG9/view?usp=drive_link)"
    )
