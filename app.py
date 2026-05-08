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

# BUNGKUS INPUT DAN TOMBOL KE DALAM SEBUAH FORM (Agar tulisan Press Enter hilang)
with st.sidebar.form(key='form_input_data'):
    link_google_sheets = st.text_input("🔗 Masukkan Link Google Sheets:", placeholder="Paste link di sini...")
    tombol_mulai = st.form_submit_button("🚀 Mulai Analisis", type="primary")

# Template Data ditaruh di luar form agar selalu terlihat
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
if tombol_mulai:
    
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
                    set1_strike, set1_dip = df['Strike_1'].dropna().astype(float).values.copy(), df['Dip_1'].dropna().astype(float).values.copy()
                    set2_strike, set2_dip = df['Strike_2'].dropna().astype(float).values.copy(), df['Dip_2'].dropna().astype(float).values.copy()
                    
                    vecs1 = np.array([strike_dip_to_pole_vector(s, d) for s, d in zip(set1_strike, set1_dip)])
                    mean_v1_norm = np.sum(vecs1, axis=0) / np.linalg.norm(np.sum(vecs1, axis=0))
                    m1_plg, m1_trd = vector_to_trend_plunge(mean_v1_norm)
                    max1_strike, max1_dip = (m1_trd + 90) % 360, 90 - m1_plg

                    vecs2 = np.array([strike_dip_to_pole_vector(s, d) for s, d in zip(set2_strike, set2_dip)])
                    mean_v2_norm = np.sum(vecs2, axis=0) / np.linalg.norm(np.sum(vecs2, axis=0))
                    m2_plg, m2_trd = vector_to_trend_plunge(mean_v2_norm)
                    max2_strike, max2_dip = (m2_trd + 90) % 360, 90 - m2_plg

                    sig2_vec = np.cross(mean_v1_norm, mean_v2_norm)
                    sig2_plg, sig2_trd = vector_to_trend_plunge(sig2_vec)

                    if np.dot(mean_v1_norm, mean_v2_norm) < 0: mean_v2_norm = -mean_v2_norm
                    sudut_bidang_deg = 180 - np.degrees(np.arccos(np.clip(np.dot(mean_v1_norm, mean_v2_norm), -1.0, 1.0)))

                    b1_plg, b1_trd = vector_to_trend_plunge(mean_v1_norm + mean_v2_norm)
                    b2_plg, b2_trd = vector_to_trend_plunge(mean_v1_norm - mean_v2_norm)

                    if sudut_bidang_deg < 90:
                        sig1_plg, sig1_trd, sig3_plg, sig3_trd = b2_plg, b2_trd, b1_plg, b1_trd
                    else:
                        sig1_plg, sig1_trd, sig3_plg, sig3_trd = b1_plg, b1_trd, b2_plg, b2_trd

                    aux_strike, aux_dip = (sig2_trd + 90) % 360, 90 - sig2_plg

                    if sig1_plg >= 60: rezim_tektonik = "Ekstensional (Tarikan)"
                    elif sig3_plg >= 60: rezim_tektonik = "Kompresional (Tekanan)"
                    elif sig2_plg >= 60: rezim_tektonik = "Wrenching (Geseran Mendatar)"
                    else: rezim_tektonik = "Campuran / Oblik"

                    with col1:
                        fig = plt.figure(figsize=(8, 8))
                        ax = fig.add_subplot(111, projection='stereonet')
                        ax.density_contourf(np.concatenate([set1_strike, set2_strike]), np.concatenate([set1_dip, set2_dip]), measurement='poles', cmap='YlOrRd')
                        ax.pole(set1_strike, set1_dip, 'k.', markersize=2, alpha=0.3)
                        ax.pole(set2_strike, set2_dip, 'k.', markersize=2, alpha=0.3)
                        ax.plane([max1_strike], [max1_dip], 'b-', linewidth=2, label='Bidang Maxima Set 1')
                        ax.plane([max2_strike], [max2_dip], 'g-', linewidth=2, label='Bidang Maxima Set 2')
                        ax.plane([aux_strike], [aux_dip], color='black', linestyle='--', linewidth=1.5, label=r'Bidang bantu ($\sigma_1-\sigma_3$)')
                        ax.line(sig1_plg, sig1_trd, marker='s', color='red', markersize=10, label=r'$\sigma_1$ (Maks)')
                        ax.line(sig2_plg, sig2_trd, marker='^', color='orange', markersize=10, label=r'$\sigma_2$ (Int)')
                        ax.line(sig3_plg, sig3_trd, marker='o', color='blue', markersize=10, label=r'$\sigma_3$ (Min)')
                        ax.grid(True)
                        plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
                        st.pyplot(fig)

                    with col2:
                        st.subheader("📊 Laporan Kinematik")
                        st.metric(label="Rezim Tektonik Dominan", value=rezim_tektonik)
                        st.write(f"**$\sigma_1$ (Maksimum):** Plunge {abs(sig1_plg):.0f}° / Trend N {sig1_trd:.0f}° E")
                        st.write(f"**$\sigma_2$ (Menengah):** Plunge {abs(sig2_plg):.0f}° / Trend N {sig2_trd:.0f}° E")
                        st.write(f"**$\sigma_3$ (Minimum):** Plunge {abs(sig3_plg):.0f}° / Trend N {sig3_trd:.0f}° E")

                # ---------------------------------------------------------
                # MODE 2: KEKAR EKSTENSI
                # ---------------------------------------------------------
                elif Mode_Analisis == "Kekar Ekstensi (Extension Joints)":
                    strike_ext, dip_ext = df['Strike_Ekstensi'].dropna().astype(float).values.copy(), df['Dip_Ekstensi'].dropna().astype(float).values.copy()
                    has_plumose = 'Pitch_Plumose' in df.columns and not df['Pitch_Plumose'].dropna().empty

                    vecs_ext = np.array([strike_dip_to_pole_vector(s, d) for s, d in zip(strike_ext, dip_ext)])
                    mean_v_ext_norm = np.sum(vecs_ext, axis=0) / np.linalg.norm(np.sum(vecs_ext, axis=0))
                    sig3_plg, sig3_trd = vector_to_trend_plunge(mean_v_ext_norm)
                    avg_ext_strike, avg_ext_dip = (sig3_trd + 90) % 360, 90 - sig3_plg

                    if has_plumose:
                        pitch_ext = df['Pitch_Plumose'].dropna().astype(float).values.copy()
                        avg_pitch = np.mean(pitch_ext)
                        sig1_plg, sig1_trd = pitch_to_trend_plunge(avg_ext_strike, avg_ext_dip, avg_pitch)
                        v_sig2 = np.cross(trend_plunge_to_vector(sig1_trd, sig1_plg), mean_v_ext_norm)
                        sig2_plg, sig2_trd = vector_to_trend_plunge(v_sig2)

                    with col1:
                        fig = plt.figure(figsize=(8, 8))
                        ax = fig.add_subplot(111, projection='stereonet')
                        ax.density_contourf(strike_ext, dip_ext, measurement='poles', cmap='Purples')
                        ax.pole(strike_ext, dip_ext, 'k.', markersize=4, alpha=0.6, label='Kutub Aktual')
                        ax.plane([avg_ext_strike], [avg_ext_dip], 'b-', linewidth=2.5, label='Bidang Ekstensi Rata-rata')
                        ax.line(sig3_plg, sig3_trd, marker='o', color='blue', markersize=10, label=r'Arah Tarikan ($\sigma_3$)')
                        
                        if has_plumose:
                            ax.line(sig1_plg, sig1_trd, marker='s', color='red', markersize=10, label=r'Rambatan Plumose ($\sigma_1$)')
                            ax.line(sig2_plg, sig2_trd, marker='^', color='orange', markersize=10, label=r'Tegasan Menengah ($\sigma_2$)')
                        ax.grid(True)
                        plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
                        st.pyplot(fig)

                    with col2:
                        st.subheader("📊 Laporan Kinematik")
                        st.write(f"**Orientasi Bidang Rata-rata:** N {avg_ext_strike:.0f}° E / {avg_ext_dip:.0f}°")
                        st.write(f"**$\sigma_3$ (Arah Tarikan):** Plunge {abs(sig3_plg):.0f}° / Trend N {sig3_trd:.0f}° E")
                        if has_plumose:
                            st.success("Tingkat Kepercayaan Data: TINGGI (Divalidasi Plumose)")
                            st.write(f"**$\sigma_1$ (Rambatan):** Plunge {abs(sig1_plg):.0f}° / Trend N {sig1_trd:.0f}° E")
                            st.write(f"**$\sigma_2$ (Menengah):** Plunge {abs(sig2_plg):.0f}° / Trend N {sig2_trd:.0f}° E")
                        else:
                            st.warning("Tingkat Kepercayaan Data: MENENGAH")
                            st.info("Hanya dapat menentukan sumbu $\sigma_3$. Tambahkan kolom 'Pitch_Plumose' untuk menentukan posisi absolut $\sigma_1$ dan $\sigma_2$.")

                # ---------------------------------------------------------
                # MODE 3: KINEMATIKA SESAR
                # ---------------------------------------------------------
                elif Mode_Analisis == "Kinematika Sesar (Fault Kinematics)":
                    strike_f, dip_f = df['Strike_Sesar'].dropna().astype(float).values.copy(), df['Dip_Sesar'].dropna().astype(float).values.copy()
                    pitch_f = df['Pitch'].dropna().astype(float).values.copy()
                    sense_f = df['Sense'].dropna().astype(str).values.copy()
                    
                    vecs_f = np.array([strike_dip_to_pole_vector(s, d) for s, d in zip(strike_f, dip_f)])
                    mean_vf_norm = np.sum(vecs_f, axis=0) / np.linalg.norm(np.sum(vecs_f, axis=0))
                    mf_plg, mf_trd = vector_to_trend_plunge(mean_vf_norm)
                    avg_strike_f, avg_dip_f = (mf_trd + 90) % 360, 90 - mf_plg
                    avg_pitch_f = np.mean(pitch_f)

                    dominant_sense = Counter([str(s).strip().lower() for s in sense_f]).most_common(1)[0][0]
                    if dominant_sense in ['normal', 'turun']: rezim_akhir = "Ekstensional (Tarikan)"
                    elif dominant_sense in ['naik', 'reverse', 'thrust']: rezim_akhir = "Kompresional (Tekanan)"
                    elif dominant_sense in ['dextral', 'sinistral', 'mendatar']: rezim_akhir = "Wrenching (Geseran Mendatar)"
                    else: rezim_akhir = "Campuran / Tidak Teridentifikasi"

                    with col1:
                        fig = plt.figure(figsize=(8, 8))
                        ax = fig.add_subplot(111, projection='stereonet')
                        ax.plane(strike_f, dip_f, 'b-', linewidth=1.5, alpha=0.3, label='Bidang Sesar Aktual')
                        ax.plane([avg_strike_f], [avg_dip_f], 'k-', linewidth=2.5, label='Bidang Sesar Rata-rata')
                        for s, d, p in zip(strike_f, dip_f, pitch_f): ax.rake(s, d, p, 'r^', markersize=8)
                        
                        handles, labels = ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        by_label['Pitch'] = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8)
                        ax.grid(True)
                        plt.legend(by_label.values(), by_label.keys(), loc='lower left', bbox_to_anchor=(1.05, 0.5))
                        st.pyplot(fig)

                    with col2:
                        st.subheader("📊 Laporan Kinematik")
                        st.metric(label="Rezim Pergerakan Dominan", value=rezim_akhir.upper())
                        st.write(f"**Total Data Sesar:** {len(strike_f)} bidang")
                        st.write(f"**Sesar Utama (Rata-rata):** N {avg_strike_f:.0f}° E / {avg_dip_f:.0f}°")
                        st.write(f"**Rata-rata Pitch (Rake):** {avg_pitch_f:.0f}°")
                        st.write(f"**Sense Dominan:** {dominant_sense.capitalize()}")

            except Exception as e:
                # Menangkap error jika link salah atau data tidak sesuai format
                st.error(f"❌ Terjadi kesalahan saat memuat data: {e}. Pastikan Link Google Sheets benar dan format kolom sesuai.")
    else:
        # Peringatan merah jika user ngeklik tombol tapi link masih kosong
        st.sidebar.error("⚠️ Link Google Sheets belum dimasukkan!")
else:
    # Tampilan default saat web baru dibuka (tombol belum ditekan)
    st.info("👈 Silakan masukkan Link Google Sheets pada panel di sebelah kiri dan klik tombol 'Mulai Analisis'.")
