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

# --- JURUS SAKTI NGILANGIN TULISAN ENTER (INJEKSI CSS) ---
st.markdown("""
    <style>
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
# ---------------------------------------------------------

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

# BUNGKUS INPUT DAN TOMBOL KE DALAM SEBUAH FORM (Agar tulisan Press Enter hilang)
with st.sidebar.form(key='form_input_data'):
    link_google_sheets = st.text_input("🔗 Masukkan Link Google Sheets:", placeholder="Paste link di sini...")
    tombol_mulai = st.form_submit_button("🚀 Mulai Analisis", type="primary")

# Template Data ditaruh di luar form agar selalu terlihat
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Template Data")
st.sidebar.info("Pilih format tabel standar sesuai dengan mode analisis Anda:")
st.sidebar.markdown("- [👉 Template Kekar Gerus Berpasangan](
