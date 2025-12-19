import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    'Meninggal', 'Jumlah Kejadian', 'Terluka', 'Menderita', 'Mengungsi',
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
    0: "Cluster 0 (Risiko Menengah)",
    1: "Cluster 1 (Risiko Rendah)",
    2: "Cluster 2 (Risiko Tinggi)",
    3: "Cluster 3 (Risiko Ekstrem)"
}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔍 Eksplorasi Data")
    menu = st.radio("Pilih Halaman:", ["Dashboard Statistik", "Cek Riwayat & Tren"])
    st.divider()
    st.caption("Data Provinsi: 2016-2020")

# --- HALAMAN 1: DASHBOARD ---
if menu == "Dashboard Statistik":
    st.title("📊 Dashboard Statistik Bencana")
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
        
        cols = ['Cluster', 'Risiko', 'Jumlah Data', 'Meninggal', 'Mengungsi', 'Rusak Berat', 'Jumlah Kejadian']
        st.dataframe(stats[cols].style.background_gradient(cmap='Reds', subset=['Meninggal', 'Mengungsi']), hide_index=True, use_container_width=True)
        
        col_pca, col_txt = st.columns([2, 1])
        with col_pca:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, ax=ax)
            st.pyplot(fig)
        with col_txt:
            st.info("*Penjelasan Kluster:*\n- Cluster 0: Kejadian sedang, dampak kerusakan menengah.\n- Cluster 1: Kejadian banyak namun dampak korban jiwa rendah.\n- Cluster 2: Dampak signifikan pada pengungsian.\n- Cluster 3: Kejadian ekstrem dengan korban jiwa/kerusakan masif.")

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
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Kejadian", f"{row['Jumlah Kejadian'].values[0]:,.0f}")
                m2.metric("Meninggal", f"{row['Meninggal'].values[0]:,.0f}")
                m3.metric("Mengungsi", f"{row['Mengungsi'].values[0]:,.0f}")
                m4.metric("Rusak Berat", f"{row['Rusak Berat'].values[0]:,.0f}")

        # Grafik Tren
        st.subheader("📈 Tren Tahunan")
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.lineplot(data=prov_data, x='Tahun', y='Cluster', marker='o', color='gray', ax=ax)
        curr = prov_data[prov_data['Tahun'] == selected_year]
        ax.scatter(curr['Tahun'], curr['Cluster'], color='red', s=200, zorder=5)
        ax.set_yticks(range(OPTIMAL_K))
        ax.set_xticks(prov_data['Tahun'].unique())
        st.pyplot(fig)


