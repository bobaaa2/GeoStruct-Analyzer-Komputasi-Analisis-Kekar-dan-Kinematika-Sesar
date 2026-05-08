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

# --- INJEKSI CSS NGILANGIN TULISAN ENTER ---
st.markdown("""
    <style>
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
# PANEL SAMPING (SIDEBAR UI)
# ==========================================
st.sidebar.header("⚙️ Panel Pengaturan")
Mode_Analisis = st.sidebar.selectbox(
    "Pilih Mode Analisis Struktur:", 
    ["Kekar Gerus Berpasangan (Conjugate Shear Joints)", 
     "Kekar Ekstensi (Extension Joints)", 
     "Kinematika Sesar (Fault Kinematics)"]
)

with st.sidebar.form(key='form_input_data'):
    link_google_sheets = st.text_input("🔗 Masukkan Link Google Sheets:", placeholder="Paste link di sini...")
    tombol_mulai = st.form_submit_button("🚀 Mulai Analisis", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Template Data")
st.sidebar.info("- [👉 Template Kekar Gerus](https://docs.google.com/spreadsheets/d/1T2296pJ1UBE5V5fpcUKeX235EbeKM3sNPJFYzohOU2g/edit?usp=drive_link)\n- [👉 Template Kekar Ekstensi](https://docs.google.com/spreadsheets/d/1qdr_HA0btAcLxe3vz_cogvLVazbU6vqNCBaVCxSlD1Q/edit?usp=drive_link)\n- [👉 Template Kinematika Sesar](https://docs.google.com/spreadsheets/d/13KssAmxUuDDf3cA9Euvr7-oJPFDK_GGPcvLCp_c79WA/edit?usp=drive_link)")
st.sidebar.warning("⚠️ Pastikan akses Google Sheets Anda **Anyone with the link**.")

# ==========================================
# LOGIKA UTAMA APLIKASI
# ==========================================
if tombol_mulai:
    if link_google_sheets != "":
        with st.spinner("Mengekstrak data..."):
            try:
                sheet_id = re.search(r'/d/([a-zA-Z0-9-_]+)', link_google_sheets).group(1)
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                df = pd.read_csv(csv_url)
                
                with st.expander("Lihat Data Mentah"):
                    st.dataframe(df, use_container_width=True)

                col1, col2 = st.columns([1.5, 1]) 

                # ---------------------------------------------------------
                # MODE 1: KEKAR GERUS
                # ---------------------------------------------------------
                if Mode_Analisis == "Kekar Gerus Berpasangan (Conjugate Shear Joints)":
                    set1_strike, set1_dip = df['Strike_1'].dropna().astype(float).values.copy(), df['Dip_1'].dropna().astype(float).values.copy()
                    set2_strike, set2_dip = df['Strike_2'].dropna().astype(float).values.copy(), df['Dip_2'].dropna().astype(float).values.copy()
                    
                    vecs1 = np.array([strike_dip_to_pole_vector(s, d) for s, d in zip(set1_strike, set1_dip)])
                    mean_v1 = np.mean(vecs1, axis=0); mean_v1 /= np.linalg.norm(mean_v1)
                    vecs2 = np.array([strike_dip_to_pole_vector(s, d) for s, d in zip(set2_strike, set2_dip)])
                    mean_v2 = np.mean(vecs2, axis=0); mean_v2 /= np.linalg.norm(mean_v2)
                    
                    if np.dot(mean_v1, mean_v2) < 0: mean_v2 = -mean_v2
                    sig2_v = np.cross(mean_v1, mean_v2); sig2_plg, sig2_trd = vector_to_trend_plunge(sig2_v)
                    
                    ang = np.degrees(np.arccos(np.clip(np.dot(mean_v1, mean_v2), -1.0, 1.0)))
                    b1_plg, b1_trd = vector_to_trend_plunge(mean_v1 + mean_v2)
                    b2_plg, b2_trd = vector_to_trend_plunge(mean_v1 - mean_v2)
                    
                    if ang < 90: sig1_plg, sig1_trd, sig3_plg, sig3_trd = b2_plg, b2_trd, b1_plg, b1_trd
                    else: sig1_plg, sig1_trd, sig3_plg, sig3_trd = b1_plg, b1_trd, b2_plg, b2_trd
                    
                    rezim = "Ekstensional" if sig1_plg >= 60 else ("Kompresional" if sig3_plg >= 60 else "Wrenching")
                    
                    with col1:
                        fig, ax = plt.subplots(subplot_kw={'projection': 'stereonet'})
                        ax.density_contourf(np.concatenate([set1_strike, set2_strike]), np.concatenate([set1_dip, set2_dip]), measurement='poles', cmap='YlOrRd', alpha=0.6)
                        ax.pole(set1_strike, set1_dip, 'k.', markersize=2, alpha=0.3)
                        ax.pole(set2_strike, set2_dip, 'k.', markersize=2, alpha=0.3)
                        # Garis Putus-putus (Movement Plane)
                        ax.plane([sig2_trd], [sig2_plg], measurement='poles', linestyle='--', color='grey', label='Movement Plane')
                        ax.line(sig1_plg, sig1_trd, 'rs', label='Sigma 1 (Maks)')
                        ax.line(sig2_plg, sig2_trd, '^', color='orange', label='Sigma 2 (Int)')
                        ax.line(sig3_plg, sig3_trd, 'bo', label='Sigma 3 (Min)')
                        plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5)); st.pyplot(fig)

                # ---------------------------------------------------------
                # MODE 2: KEKAR EKSTENSI
                # ---------------------------------------------------------
                elif Mode_Analisis == "Kekar Ekstensi (Extension Joints)":
                    s_e, d_e = df['Strike_Ekstensi'].dropna().astype(float).values.copy(), df['Dip_Ekstensi'].dropna().astype(float).values.copy()
                    has_plumose = 'Pitch_Plumose' in df.columns and not df['Pitch_Plumose'].dropna().empty
                    
                    v_e = np.array([strike_dip_to_pole_vector(s, d) for s, d in zip(s_e, d_e)])
                    m_e = np.mean(v_e, axis=0); m_e /= np.linalg.norm(m_e)
                    s3_plg, s3_trd = vector_to_trend_plunge(m_e)
                    
                    with col1: 
                        fig, ax = plt.subplots(subplot_kw={'projection': 'stereonet'})
                        ax.pole(s_e, d_e, 'k.', markersize=4, alpha=0.5)
                        ax.line(s3_plg, s3_trd, 'bo', markersize=10, label='Sigma 3 (Min)')
                        
                        if has_plumose:
                            p_p = df['Pitch_Plumose'].dropna().astype(float).values.copy()
                            avg_s, avg_d = (s3_trd + 90) % 360, 90 - s3_plg
                            s1_plg, s1_trd = pitch_to_trend_plunge(avg_s, avg_d, np.mean(p_p))
                            v2 = np.cross(trend_plunge_to_vector(s1_trd, s1_plg), m_e)
                            s2_plg, s2_trd = vector_to_trend_plunge(v2)
                            # Garis Putus-putus
                            ax.plane([s2_trd], [s2_plg], measurement='poles', linestyle='--', color='grey')
                            ax.line(s1_plg, s1_trd, 'rs', markersize=10, label='Sigma 1 (Maks)')
                            ax.line(s2_plg, s2_trd, '^', color='orange', markersize=10, label='Sigma 2 (Int)')
                        
                        plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5)); st.pyplot(fig)

                # ---------------------------------------------------------
                # MODE 3: KINEMATIKA SESAR
                # ---------------------------------------------------------
                elif Mode_Analisis == "Kinematika Sesar (Fault Kinematics)":
                    s_f, d_f = df['Strike_Sesar'].dropna().astype(float).values.copy(), df['Dip_Sesar'].dropna().astype(float).values.copy()
                    p_f, sense_f = df['Pitch'].dropna().astype(float).values.copy(), df['Sense'].dropna().astype(str).values.copy()
                    
                    p_vecs, t_vecs, b_vecs = [], [], []
                    for s, d, p, sense in zip(s_f, d_f, p_f, sense_f):
                        n = strike_dip_to_pole_vector(s, d)
                        plg_s, trd_s = pitch_to_trend_plunge(s, d, p)
                        slk = trend_plunge_to_vector(trd_s, plg_s)
                        
                        if any(x in sense.lower() for x in ['naik', 'reverse', 'thrust']):
                            p_axis = slk + n; t_axis = slk - n
                        else:
                            p_axis = slk - n; t_axis = slk + n
                        
                        p_axis /= np.linalg.norm(p_axis); t_axis /= np.linalg.norm(t_axis)
                        b_axis = np.cross(p_axis, t_axis); b_axis /= np.linalg.norm(b_axis)
                        p_vecs.append(p_axis); t_vecs.append(t_axis); b_vecs.append(b_axis)

                    s1_v = np.mean(p_vecs, axis=0); s1_v /= np.linalg.norm(s1_v); s1_plg, s1_trd = vector_to_trend_plunge(s1_v)
                    s3_v = np.mean(t_vecs, axis=0); s3_v /= np.linalg.norm(s3_v); s3_plg, s3_trd = vector_to_trend_plunge(s3_v)
                    s2_v = np.mean(b_vecs, axis=0); s2_v /= np.linalg.norm(s2_v); s2_plg, s2_trd = vector_to_trend_plunge(s2_v)

                    with col1:
                        fig, ax = plt.subplots(subplot_kw={'projection': 'stereonet'})
                        ax.plane(s_f, d_f, 'b-', alpha=0.15)
                        for s, d, p in zip(s_f, d_f, p_f): ax.rake(s, d, p, 'r^', markersize=5, alpha=0.4)
                        # Garis Putus-putus (Movement Plane)
                        ax.plane([s2_trd], [s2_plg], measurement='poles', linestyle='--', color='grey', label='Movement Plane')
                        ax.line(s1_plg, s1_trd, 'rs', markersize=10, label='Sigma 1 (P)')
                        ax.line(s2_plg, s2_trd, '^', color='orange', markersize=10, label='Sigma 2 (B)')
                        ax.line(s3_plg, s3_trd, 'bo', markersize=10, label='Sigma 3 (T)')
                        plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5)); ax.grid(True); st.pyplot(fig)

            except Exception as e: st.error(f"Error: {e}")
    else: st.sidebar.error("Link kosong!")
