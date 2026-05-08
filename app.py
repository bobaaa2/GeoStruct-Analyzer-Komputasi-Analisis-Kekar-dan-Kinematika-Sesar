import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
import re
from collections import Counter

# ==========================================
# KONFIGURASI HALAMAN WEB
# ==========================================
st.set_page_config(page_title="GeoStruct Analyzer", page_icon="🧭", layout="wide")
st.title("🧭 GeoStruct Analyzer")
st.markdown("Aplikasi Komputasi Kinematika Struktur Geologi Berbasis Vektor 3D")
st.markdown("---")

# ==========================================
# FUNGSI KOMPUTASI VEKTOR
# ==========================================
def strike_dip_to_pole_vector(strike, dip):
    trd = np.radians((strike - 90) % 360)
    plg = np.radians(90 - dip)
    return np.array([np.cos(trd) * np.cos(plg), np.sin(trd) * np.cos(plg), np.sin(plg)])

def vector_to_trend_plunge(v):
    if v[2] < 0: v = -v
    v = v / np.linalg.norm(v)
    plg = np.degrees(np.arcsin(v[2]))
    trd = np.degrees(np.arctan2(v[1], v[0])) % 360
    return plg, trd

def pitch_to_trend_plunge(strike, dip, pitch):
    s_rad = np.radians(strike)
    d_rad = np.radians(dip)
    p_rad = np.radians(pitch)
    plg = np.degrees(np.arcsin(np.sin(d_rad) * np.sin(p_rad)))
    beta = np.degrees(np.arctan(np.cos(d_rad) * np.tan(p_rad)))
    trd = (strike + beta) % 360
    return plg, trd

def trend_plunge_to_vector(trend, plunge):
    trd = np.radians(trend)
    plg = np.radians(plunge)
    return np.array([np.cos(trd) * np.cos(plg), np.sin(trd) * np.cos(plg), np.sin(plg)])

# ==========================================
# PANEL SAMPING (SIDEBAR UI) & TEMPLATE DATA
# ==========================================
st.sidebar.header("⚙️ Panel Pengaturan")
Mode_Analisis = st.sidebar.selectbox(
    "Pilih Mode Analisis Struktur:", 
    ["Kekar Gerus Berpasangan (Conjugate Shear Joints)", 
     "Kekar Ekstensi (Extension Joints)", 
     "Kinematika Sesar (Fault Kinematics)"]
)

link_google_sheets = st.sidebar.text_input("🔗 Masukkan Link Google Sheets:", placeholder="Paste link di sini...")

# Template ditaruh DI LUAR tombol agar selalu terlihat oleh user sejak awal web dibuka
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Template Data")
st.sidebar.info("Pilih format tabel standar sesuai dengan mode analisis Anda:")
st.sidebar.markdown("- [👉 Template Kekar Gerus Berpasangan](https://docs.google.com/spreadsheets/d/1T2296pJ1UBE5V5fpcUKeX235EbeKM3sNPJFYzohOU2g/edit?usp=drive_link)")
st.sidebar.markdown("- [👉 Template Kekar Ekstensi (Mode I)](https://docs.google.com/spreadsheets/d/1qdr_HA0btAcLxe3vz_cogvLVazbU6vqNCBaVCxSlD1Q/edit?usp=drive_link)")
st.sidebar.markdown("- [👉 Template Kinematika Sesar](https://docs.google.com/spreadsheets/d/13KssAmxUuDDf3cA9Euvr7-oJPFDK_GGPcvLCp_c79WA/edit?usp=drive_link)")
st.sidebar.caption("💡 **Cara pakai:** Buka link -> Klik menu **File** -> **Make a copy** (Buat salinan) ke Drive Anda -> Isi dengan data pengukuran lapangan Anda.")
st.sidebar.warning("⚠️ **PENTING!!!:**\nPastikan akses salinan Google Sheets Anda **TIDAK PRIVATE**. Ubah pengaturan Share/Bagikan menjadi **'Anyone with the link'** (Siapa saja yang memiliki link) agar mesin dapat membaca datanya.")
st.sidebar.markdown("---")

# ==========================================
# LOGIKA UTAMA APLIKASI (DIJALANKAN OLEH TOMBOL)
# ==========================================
if st.sidebar.button("🚀 Mulai Analisis", type="primary"):
    
    # 1. Validasi: Cek apakah user sudah mengisi link
    if link_google_sheets != "":
        
        # 2. Aktifkan animasi loading
        with st.spinner("Mengekstrak data dari server Google..."):
            try:
                # 3. Penarikan Data dari Google Sheets
                sheet_id = re.search(r'/d/([a-zA-Z0-9-_]+)', link_google_sheets).group(1)
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                df = pd.read_csv(csv_url)
                st.sidebar.success("✅ Data berhasil diimpor!")
                
                # 4. Tampilkan tabel mentah untuk pengecekan
                with st.expander("Lihat Data Mentah (Spreadsheet)"):
                    st.dataframe(df, use_container_width=True)

                # 5. Persiapan Layout (Kiri = Stereonet, Kanan = Teks Laporan)
                col1, col2 = st.columns([1.5, 1]) 

                # ---------------------------------------------------------
                # MODE 1: KEKAR GERUS
                # ---------------------------------------------------------
                if Mode_Analisis == "Kekar Gerus Berpasangan (Conjugate Shear Joints)":
                    set1_strike, set1_dip = df['Strike_1'].dropna().astype(float).values.copy(), df['Dip_1'].dropna().astype(
