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

# ==========================================
# LOGIKA UTAMA APLIKASI
# ==========================================
if tombol_mulai:
    if link_google_sheets != "":
        with st.spinner("Mengekstrak data dan menghitung..."):
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
                    s1, d1 = df['Strike_1'].dropna().astype(float).values.copy(), df['Dip_1'].dropna().astype(float).values.copy()
                    s2, d2 = df['Strike_2'].dropna().astype(float).values.copy(), df['Dip_2'].dropna().astype(float).values.copy()
                    v1 = np.array([strike_dip_to_pole_vector(s, d) for s, d in zip(s1, d1)])
                    m1 = np.mean(v1, axis=0); m1 /= np.linalg.norm(m1)
                    v2 = np.array([strike_dip_to_pole_vector(s, d) for s, d in zip(s2, d2)])
                    m2 = np.mean(v2, axis=0); m2 /= np.linalg.norm(m2)
                    if np.dot(m1, m2) < 0: m2 = -m2
                    sig2_v = np.cross(m1, m2); sig2_plg, sig2_trd = vector_to_trend_plunge(sig2_v)
                    ang = np.degrees(np.arccos(np.clip(np.dot(m1, m2), -1.0, 1.0)))
                    b1_plg, b1_trd = vector_to_trend_plunge(m1 + m2)
                    b2_plg, b2_trd = vector_to_trend_plunge(m1 - m2)
                    if ang < 90: sig1_plg, sig1_trd, sig3_plg, sig3_trd = b2_plg, b2_trd, b1_plg, b1_trd
                    else: sig1_plg, sig1_trd, sig3_plg, sig3_trd = b1_plg, b1_trd, b2_plg, b2_trd
                    rezim = "EKSTENSIONAL" if sig1_plg >= 60 else ("KOMPRESIONAL" if sig3_plg >= 60 else "WRENCHING (MENDATAR)")
                    
                    with col1:
                        fig, ax = plt.subplots(subplot_kw={'projection': 'stereonet'})
                        ax.pole(s1, d1, 'k.', alpha=0.3); ax.pole(s2, d2, 'k.', alpha=0.3)
                        ax.plane([(sig2_trd + 90) % 360], [90 - sig2_plg], linestyle='--', color='grey', label='Movement Plane')
                        ax.line(sig1_plg, sig1_trd, 'rs', markersize=10, label='Sigma 1')
                        ax.line(sig2_plg, sig2_trd, '^', color='orange', markersize=10, label='Sigma 2')
                        ax.line(sig3_plg, sig3_trd, 'bo', markersize=10, label='Sigma 3')
                        # LEGEND DIKECILIN DISINI
                        plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5), prop={'size': 5}); ax.grid(True); st.pyplot(fig)
                    with col2:
                        st.subheader("📊 Analisis Kekar Gerus")
                        st.write("Rezim Tektonik Dominan:")
                        st.success(f"**{rezim}**")
                        st.write(f"**$\sigma_1$:** {sig1_plg:.0f}° / {sig1_trd:.0f}°")
                        st.write(f"**$\sigma_2$:** {sig2_plg:.0f}° / {sig2_trd:.0f}°")
                        st.write(f"**$\sigma_3$:** {sig3_plg:.0f}° / {sig3_trd:.0f}°")

                # ---------------------------------------------------------
                # MODE 2: KEKAR EKSTENSI
                # ---------------------------------------------------------
                elif Mode_Analisis == "Kekar Ekstensi (Extension Joints)":
                    s_e, d_e = df['Strike_Ekstensi'].dropna().astype(float).values.copy(), df['Dip_Ekstensi'].dropna().astype(float).values.copy()
                    v_e = np.array([strike_dip_to_pole_vector(s, d) for s, d in zip(s_e, d_e)])
                    m_e = np.mean(v_e, axis=0); m_e /= np.linalg.norm(m_e); s3_plg, s3_trd = vector_to_trend_plunge(m_e)
                    with col1:
                        fig, ax = plt.subplots(subplot_kw={'projection': 'stereonet'})
                        ax.pole(s_e, d_e, 'k.', alpha=0.5); ax.line(s3_plg, s3_trd, 'bo', markersize=10, label='Sigma 3')
                        if 'Pitch_Plumose' in df.columns and not df['Pitch_Plumose'].dropna().empty:
                            p_p = df['Pitch_Plumose'].dropna().astype(float).values.copy()
                            s1_plg, s1_trd = pitch_to_trend_plunge((s3_trd+90)%360, 90-s3_plg, np.mean(p_p))
                            v2 = np.cross(trend_plunge_to_vector(s1_trd, s1_plg), m_e); s2_plg, s2_trd = vector_to_trend_plunge(v2)
                            ax.plane([(s2_trd + 90) % 360], [90 - s2_plg], linestyle='--', color='grey', label='Movement Plane')
                            ax.line(s1_plg, s1_trd, 'rs', label='Sigma 1'); ax.line(s2_plg, s2_trd, '^', color='orange', label='Sigma 2')
                        # LEGEND DIKECILIN DISINI
                        plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5), prop={'size': 8}); ax.grid(True); st.pyplot(fig)
                    with col2:
                        st.subheader("📊 Analisis Kekar Ekstensi")
                        st.write(f"**$\sigma_3$ (Arah Tarikan):** {s3_plg:.0f}° / {s3_trd:.0f}°")

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
                        if any(x in sense.lower() for x in ['naik', 'reverse', 'thrust']): p_axis = slk + n; t_axis = slk - n
                        else: p_axis = slk - n; t_axis = slk + n
                        p_axis /= np.linalg.norm(p_axis); t_axis /= np.linalg.norm(t_axis); b_axis = np.cross(p_axis, t_axis); b_axis /= np.linalg.norm(b_axis)
                        p_vecs.append(p_axis); t_vecs.append(t_axis); b_vecs.append(b_axis)
                    s1_v = np.mean(p_vecs, axis=0); s1_v /= np.linalg.norm(s1_v); s1_plg, s1_trd = vector_to_trend_plunge(s1_v)
                    s3_v = np.mean(t_vecs, axis=0); s3_v /= np.linalg.norm(s3_v); s3_plg, s3_trd = vector_to_trend_plunge(s3_v)
                    s2_v = np.mean(b_vecs, axis=0); s2_v /= np.linalg.norm(s2_v); s2_plg, s2_trd = vector_to_trend_plunge(s2_v)
                    if s1_plg >= 60: rezim = "EKSTENSIONAL (SESAR NORMAL)"
                    elif s3_plg >= 60: rezim = "KOMPRESIONAL (SESAR NAIK)"
                    elif s2_plg >= 60: rezim = "WRENCHING (SESAR MENDATAR)"
                    else: rezim = "REZIM OBLIK (CAMPURAN)"

                    with col1:
                        fig, ax = plt.subplots(subplot_kw={'projection': 'stereonet'})
                        ax.plane(s_f, d_f, 'b-', alpha=0.15)
                        ax.plane([(s2_trd + 90) % 360], [90 - s2_plg], linestyle='--', color='grey', label='Movement Plane')
                        ax.line(s1_plg, s1_trd, 'rs', markersize=10, label='Sigma 1 (P)')
                        ax.line(s2_plg, s2_trd, '^', color='orange', markersize=10, label='Sigma 2 (B)')
                        ax.line(s3_plg, s3_trd, 'bo', markersize=10, label='Sigma 3 (T)')
                        # LEGEND DIKECILIN DISINI
                        plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5), prop={'size': 8}); ax.grid(True); st.pyplot(fig)
                    with col2:
                        st.subheader("📊 Analisis Sesar")
                        st.write("Rezim Dominan:")
                        st.success(f"**{rezim}**")
                        st.write(f"**$\sigma_1$:** {s1_plg:.0f}° / {s1_trd:.0f}°")
                        st.write(f"**$\sigma_2$:** {s2_plg:.0f}° / {s2_trd:.0f}°")
                        st.write(f"**$\sigma_3$:** {s3_plg:.0f}° / {s3_trd:.0f}°")

            except Exception as e: st.error(f"Error: {e}")
    else: st.sidebar.error("Link Google Sheets kosong!")
