import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="KLASIKA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stSelectbox label { font-weight: bold; font-size: 1.1em; }
    </style>
""", unsafe_allow_html=True)

# --- DAFTAR PROVINSI TARGET (33 Provinsi) ---
DAFTAR_PROVINSI_TARGET = [
    "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Jambi", 
    "Sumatera Selatan", "Bengkulu", "Lampung", "Kepulauan Bangka Belitung", 
    "Kepulauan Riau", "DKI Jakarta", "Jawa Barat", "Jawa Tengah", 
    "Daerah Istimewa Yogyakarta", "Jawa Timur", "Banten", "Bali", 
    "Nusa Tenggara Timur", "Nusa Tenggara Barat", "Kalimantan Barat", 
    "Kalimantan Tengah", "Kalimantan Selatan", "Kalimatan Timur", 
    "Kalimantan Utara", "Sulawesi Utara", "Sulawesi Tengah", 
    "Sulawesi Selatan", "Sulawesi Tenggara", "Gorontalo", 
    "Sulawesi Barat", "Maluku", "Maluku Utara", "Papua"
]

# --- FILE CONFIG ---
FILE_TOTAL = "DampakBencanaFinal.csv"      
FILE_TAHUNAN = "provinsi_rawan_bencana.csv" 

FEATURES = [
    'Jumlah Kejadian', 'Meninggal',  'Terluka', 'Menderita', 'Mengungsi',
    'Rusak Berat', 'Rusak Sedang', 'Rusak Ringan', 'Terendam'
]
OPTIMAL_K = 4

# --- FUNGSI PEMBERSIH NAMA PROVINSI ---
def normalize_prov_name(raw_name):
    """Membersihkan nama provinsi agar sesuai dengan daftar target."""
    name = str(raw_name).replace("Bencana Menurut Wilayah", "").strip().title()
    
    corrections = {
        "Dki Jakarta": "DKI Jakarta",
        "Di Yogyakarta": "Daerah Istimewa Yogyakarta",
        "D.I. Yogyakarta": "Daerah Istimewa Yogyakarta",
        "Nusa Tenggara Barat": "Nusa Tenggara Barat",
        "Nusa Tenggara Timur": "Nusa Tenggara Timur",
        # Koreksi jika ada typo di CSV
        "Kalimantan Timur": "Kalimatan Timur" if "Kalimatan Timur" in DAFTAR_PROVINSI_TARGET else "Kalimantan Timur"
    }
    return corrections.get(name, name)

# --- LOAD DATA TOTAL ---
@st.cache_data
def load_data_total():
    try:
        df = pd.read_csv(FILE_TOTAL, delimiter=";")
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if col not in ['Wilayah', 'No.']:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False), errors='coerce').fillna(0)
        return df
    except Exception as e:
        return pd.DataFrame()

# --- LOAD DATA TAHUNAN (SCANNING GRID) ---
@st.cache_data
def load_data_tahunan():
    try:
        # Baca tanpa header agar bisa scan seluruh grid
        df_raw = pd.read_csv(FILE_TAHUNAN, header=None, low_memory=False)
        all_data = []

        # LOGIKA SCANNING: Cari koordinat "No." di seluruh file
        for r in range(df_raw.shape[0]):
            for c in range(df_raw.shape[1]):
                val = str(df_raw.iloc[r, c]).strip()
                
                if val == "No.":
                    # Cek baris di atasnya untuk Nama Provinsi
                    if r > 0:
                        raw_prov = df_raw.iloc[r-1, c]
                        clean_prov = normalize_prov_name(raw_prov)
                        
                        # Ambil blok data (10 baris ke bawah, 11 kolom ke kanan)
                        end_r = min(r + 10, df_raw.shape[0])
                        block = df_raw.iloc[r+1:end_r, c:c+11].copy()
                        
                        block.columns = ['No.', 'Tahun', 'Meninggal', 'Jumlah Kejadian', 'Terluka', 
                                         'Menderita', 'Mengungsi', 'Rusak Berat', 'Rusak Sedang', 
                                         'Rusak Ringan', 'Terendam']
                        
                        block['Provinsi'] = clean_prov
                        
                        # Filter baris yang valid (No dan Tahun harus angka)
                        block = block[pd.to_numeric(block['No.'], errors='coerce').notna()]
                        block = block[pd.to_numeric(block['Tahun'], errors='coerce').notna()]
                        
                        if not block.empty:
                            all_data.append(block)

        if not all_data: return pd.DataFrame()
        
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Cleaning Angka
        for col in FEATURES + ['Tahun']:
            final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace('.', '', regex=False), errors='coerce').fillna(0)
        
        final_df['Tahun'] = final_df['Tahun'].astype(int)
        
        return final_df

    except Exception as e:
        st.error(f"Error membaca file tahunan: {e}")
        return pd.DataFrame()

# --- PREPARE DATA ---
df_total = load_data_total()
df_tahunan = load_data_tahunan()

# Training Model
@st.cache_resource
def train_model(df):
    if df.empty: return None, None, None, None
    X = df[FEATURES]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(scaled)
    
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df['Cluster']
    return df, scaler, kmeans, pca_df

df_model, scaler, model, pca_df = train_model(df_tahunan.copy())

RISK_LEVEL = {
    0: "Cluster 0 (Risiko Rendah)",
    1: "Cluster 1 (Risiko Menengah)",
    2: "Cluster 2 (Risiko Tinggi)",
    3: "Cluster 3 (Risiko Ekstrem)"
}

# Koordinat sederhana untuk pemetaan (Pusat Provinsi)
PROV_COORDS = {
    "Aceh": {"lat": 4.6951, "lon": 96.7494}, "Sumatera Utara": {"lat": 2.1121, "lon": 99.3982},
    "Sumatera Barat": {"lat": -0.7399, "lon": 100.8000}, "Riau": {"lat": 0.2933, "lon": 101.7068},
    "Jambi": {"lat": -1.6101, "lon": 103.6131}, "Sumatera Selatan": {"lat": -3.3194, "lon": 104.9145},
    "Bengkulu": {"lat": -3.7928, "lon": 102.2608}, "Lampung": {"lat": -4.5586, "lon": 105.4068},
    "Kepulauan Bangka Belitung": {"lat": -2.7410, "lon": 106.4406}, "Kepulauan Riau": {"lat": 3.9456, "lon": 108.1429},
    "DKI Jakarta": {"lat": -6.2088, "lon": 106.8456}, "Jawa Barat": {"lat": -7.0909, "lon": 107.6689},
    "Jawa Tengah": {"lat": -7.1510, "lon": 110.1403}, "Daerah Istimewa Yogyakarta": {"lat": -7.8753, "lon": 110.4262},
    "Jawa Timur": {"lat": -7.5361, "lon": 112.2384}, "Banten": {"lat": -6.4058, "lon": 106.0640},
    "Bali": {"lat": -8.4095, "lon": 115.1889}, "Nusa Tenggara Barat": {"lat": -8.6529, "lon": 117.3616},
    "Nusa Tenggara Timur": {"lat": -8.6574, "lon": 121.0794}, "Kalimantan Barat": {"lat": -0.2787, "lon": 109.9754},
    "Kalimantan Tengah": {"lat": -1.6815, "lon": 113.3824}, "Kalimantan Selatan": {"lat": -3.0926, "lon": 115.2838},
    "Kalimatan Timur": {"lat": 0.4538, "lon": 116.2420}, "Kalimantan Utara": {"lat": 3.0731, "lon": 116.0414},
    "Sulawesi Utara": {"lat": 0.6247, "lon": 123.9750}, "Sulawesi Tengah": {"lat": -1.4300, "lon": 121.4456},
    "Sulawesi Selatan": {"lat": -3.6688, "lon": 119.9740}, "Sulawesi Tenggara": {"lat": -4.1449, "lon": 122.1746},
    "Gorontalo": {"lat": 0.6999, "lon": 122.4467}, "Sulawesi Barat": {"lat": -2.8440, "lon": 119.2321},
    "Maluku": {"lat": -3.2385, "lon": 130.1453}, "Maluku Utara": {"lat": 1.5700, "lon": 127.8000},
    "Papua": {"lat": -4.2699, "lon": 138.0804}
}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔍 Eksplorasi Data")
    menu = st.radio("Pilih Halaman:", ["Statistik Bencana", "Cek Riwayat & Tren"])
    st.divider()
    st.caption("Data Provinsi: 2016-2020")

# --- HALAMAN 1: DASHBOARD ---
if menu == "Statistik Bencana":
    st.title("📊 KLASIKA: Klasterisasi Risiko Bencana")
    st.markdown("Sumber Data diperoleh dari Badan Nasional Penanggulangan Bencana (BNPB)")
    
    c1, c2, c3, c4 = st.columns(4)
    if not df_total.empty:
        c1.metric("Total Kejadian", f"{df_total['Jumlah Kejadian'].sum():,.0f}")
        c2.metric("Meninggal", f"{df_total['Meninggal'].sum():,.0f}")
        c3.metric("Mengungsi", f"{df_total['Mengungsi'].sum():,.0f}")
        c4.metric("Rusak Berat", f"{df_total['Rusak Berat'].sum():,.0f}")

    st.divider()
    st.subheader("📋 Statistik Karakteristik Kluster")
    
    if df_model is not None and not df_model.empty:
        stats = df_model.groupby('Cluster')[FEATURES].mean().reset_index()
        stats['Jumlah Data'] = df_model['Cluster'].value_counts().sort_index().values
        stats['Risiko'] = stats['Cluster'].map(RISK_LEVEL)
        
        cols = ['Cluster', 'Risiko', 'Jumlah Data', 'Jumlah Kejadian', 'Meninggal', 'Terluka', 'Menderita', 'Mengungsi', 'Rusak Berat', 'Rusak Sedang', 'Rusak Ringan', 'Terendam']
        st.dataframe(stats[cols].round(2), hide_index=True, use_container_width=True)
        
        # --- TAMBAHAN VISUALISASI PETA INDONESIA ---
        st.subheader("🗺️ Sebaran Risiko Berdasarkan Wilayah")
        
        # Siapkan data untuk peta
        map_df = df_model.groupby('Provinsi')['Cluster'].last().reset_index() # Ambil status cluster terakhir
        map_df['lat'] = map_df['Provinsi'].map(lambda x: PROV_COORDS.get(x, {}).get('lat'))
        map_df['lon'] = map_df['Provinsi'].map(lambda x: PROV_COORDS.get(x, {}).get('lon'))
        map_df['Risiko'] = map_df['Cluster'].map(RISK_LEVEL)
        
        # Hapus data yang koordinatnya tidak ditemukan
        map_df = map_df.dropna(subset=['lat', 'lon'])

        fig_map = px.scatter_mapbox(
            map_df, lat="lat", lon="lon", color="Cluster",
            size=map_df['Cluster'] + 2, # Ukuran titik
            color_continuous_scale="Reds",
            hover_name="Provinsi",
            hover_data={"Risiko": True, "Cluster": True, "lat": False, "lon": False},
            zoom=3.5, center={"lat": -2.5, "lon": 118},
            mapbox_style="carto-positron", height=500
        )
        st.plotly_chart(fig_map, use_container_width=True)
        # --- END TAMBAHAN PETA ---

        col_pca, col_txt = st.columns([2, 1])
        with col_pca:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, ax=ax)
            st.pyplot(fig)
        with col_txt:
            st.info("*Penjelasan Kluster:*\n- Cluster 0:  Kejadian dengan intensitas sedang dan dampak kerusakan menengah.\n- Cluster 1: Kejadian yang sering terjadi namun dengan angka fatalitas (korban jiwa) yang relatif rendah.\n- Cluster 2: Kejadian yang memiliki dampak pengungsian sangat signifikan.\n- Cluster 3: Kejadian ekstrem yang mengakibatkan korban jiwa massal atau kerusakan infrastruktur yang sangat masif.")

# --- HALAMAN 2: CEK RIWAYAT (SCROLLDOWN) ---
elif menu == "Cek Riwayat & Tren":
    st.title("📅 Cek Riwayat & Tren Risiko")
    
    if df_model is None or df_model.empty:
        st.error("Data Tahunan gagal dimuat. Cek file CSV.")
        st.stop()

    # --- MENU SELECTBOX ---
    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        # Gunakan list target, pastikan ada di data
        avail = sorted(df_model['Provinsi'].unique())
        # Gabungkan: Prioritas dari DAFTAR_TARGET, sisanya dari Data (jika ada)
        final_list = [p for p in DAFTAR_PROVINSI_TARGET if p in avail]
        
        selected_prov = st.selectbox("1️⃣ Pilih Provinsi:", final_list)
        
    with c_sel2:
        prov_data = df_model[df_model['Provinsi'] == selected_prov].sort_values('Tahun')
        if not prov_data.empty:
            years = sorted(prov_data['Tahun'].unique())
            selected_year = st.selectbox("2️⃣ Pilih Tahun:", years)
        else:
            selected_year = None
            st.warning("⚠ Data untuk provinsi ini tidak ditemukan dalam file.")

    st.divider()

    if selected_year:
        row = prov_data[prov_data['Tahun'] == selected_year]
        if not row.empty:
            clust = row['Cluster'].values[0]
            
            # Tampilan Hasil
            st.subheader(f"Hasil: {selected_prov} ({selected_year})")
            cr1, cr2 = st.columns([1, 3])
            with cr1:
                color = {0:"#28a745", 1:"#ffc107", 2:"#fd7e14", 3:"#dc3545"}.get(clust, "#333")
                st.markdown(f"<h1 style='text-align:center; color:{color}; font-size: 80px; margin:0;'>{clust}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center; font-weight:bold;'>{RISK_LEVEL[clust]}</p>", unsafe_allow_html=True)
            with cr2:
                # Mengambil hanya kolom fitur untuk ditampilkan di tabel
                display_row = row[FEATURES].copy()
    
                # Menampilkan dataframe agar rapi seperti bagian statistik kluster
                st.dataframe(
                    display_row.round(2), 
                    hide_index=True, 
                    use_container_width=True
                )

        # Grafik Tren
        st.subheader("📈 Tren Tahunan")
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.lineplot(data=prov_data, x='Tahun', y='Cluster', marker='o', color='gray', ax=ax)
        curr = prov_data[prov_data['Tahun'] == selected_year]
        ax.scatter(curr['Tahun'], curr['Cluster'], color='red', s=200, zorder=5)
        ax.set_yticks(range(OPTIMAL_K))
        ax.set_xticks(prov_data['Tahun'].unique())
        st.pyplot(fig)


