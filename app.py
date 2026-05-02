import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="Jaya Jaya Institut - Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white; padding: 2rem 2.5rem; border-radius: 16px;
        margin-bottom: 2rem; text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08); border-left: 5px solid #0f3460;
        margin-bottom: 1rem;
    }
    .metric-card h3 { font-size: 0.85rem; color: #666; margin: 0; font-weight: 600; text-transform: uppercase; }
    .metric-card p  { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; margin: 0.3rem 0 0 0; }
    .result-box-dropout  { background:#ffeaea; border:2px solid #E74C3C; border-radius:12px; padding:1.5rem; text-align:center; }
    .result-box-graduate { background:#eafff2; border:2px solid #2ECC71; border-radius:12px; padding:1.5rem; text-align:center; }
    .result-box-enrolled { background:#eaf4ff; border:2px solid #3498DB; border-radius:12px; padding:1.5rem; text-align:center; }
    .result-box-dropout h2  { color:#E74C3C; }
    .result-box-graduate h2 { color:#2ECC71; }
    .result-box-enrolled h2 { color:#3498DB; }
    .stButton>button {
        background: linear-gradient(135deg, #0f3460, #533483);
        color: white; border: none; border-radius: 8px;
        padding: 0.7rem 2rem; font-weight: 600; font-size: 1rem; width: 100%;
        transition: transform 0.2s; cursor: pointer;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(15,52,96,0.4); }
    .sidebar-title { font-size:1.3rem; font-weight:700; color:#0f3460; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_dir = Path("model")
    try:
        model = joblib.load(model_dir / "best_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        le = joblib.load(model_dir / "label_encoder.pkl")
        features = joblib.load(model_dir / "feature_names.pkl")
        return model, scaler, le, features
    except FileNotFoundError:
        return None, None, None, None


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data .csv", sep=";")
    except Exception:
        return None


model, scaler, le, feature_names = load_model()
df = load_data()

st.markdown("""
<div class="main-header">
    <h1>🎓 Jaya Jaya Institut</h1>
    <p>Student Performance & Dropout Prediction System</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📊 Dashboard", "🤖 Prediksi", "📈 Analisis Data"])

# ── Tab 1 : Dashboard ─────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("📊 Overview Performa Mahasiswa")
    if df is not None:
        total = len(df)
        n_dropout  = (df["Status"] == "Dropout").sum()
        n_graduate = (df["Status"] == "Graduate").sum()
        n_enrolled = (df["Status"] == "Enrolled").sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Mahasiswa", f"{total:,}")
        with c2:
            st.metric("🔴 Dropout", f"{n_dropout:,}", f"{n_dropout/total*100:.1f}%")
        with c3:
            st.metric("🟢 Graduate", f"{n_graduate:,}", f"{n_graduate/total*100:.1f}%")
        with c4:
            st.metric("🔵 Enrolled", f"{n_enrolled:,}", f"{n_enrolled/total*100:.1f}%")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#E74C3C", "#2ECC71", "#3498DB"]
            counts = df["Status"].value_counts()
            wedges, texts, autotexts = ax.pie(
                counts, labels=counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2}
            )
            for at in autotexts:
                at.set_fontsize(10); at.set_fontweight("bold")
            ax.set_title("Distribusi Status Mahasiswa", fontsize=12, fontweight="bold")
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            grad = df[df["Status"] == "Graduate"]
            drop = df[df["Status"] == "Dropout"]
            ax.hist(grad["Curricular_units_2nd_sem_grade"], bins=25, alpha=0.65,
                    label="Graduate", color="#2ECC71", edgecolor="white")
            ax.hist(drop["Curricular_units_2nd_sem_grade"], bins=25, alpha=0.65,
                    label="Dropout", color="#E74C3C", edgecolor="white")
            ax.set_title("Distribusi Nilai Sem-2: Graduate vs Dropout", fontsize=11, fontweight="bold")
            ax.set_xlabel("Nilai")
            ax.set_ylabel("Frekuensi")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            scholar_status = df.groupby(["Scholarship_holder", "Status"]).size().unstack(fill_value=0)
            scholar_status.index = ["Non-Beasiswa", "Beasiswa"]
            scholar_status.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.55)
            ax.set_title("Status vs Beasiswa", fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title="Status")
            st.pyplot(fig)
            plt.close()

        with col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            tuition_status = df.groupby(["Tuition_fees_up_to_date", "Status"]).size().unstack(fill_value=0)
            tuition_status.index = ["Menunggak", "Lunas"]
            tuition_status.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.55)
            ax.set_title("Status vs Ketepatan Pembayaran SPP", fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title="Status")
            st.pyplot(fig)
            plt.close()
    else:
        st.error("Dataset tidak ditemukan. Pastikan file 'data .csv' tersedia.")

# ── Tab 2 : Prediksi ──────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🤖 Prediksi Status Mahasiswa")
    if model is None:
        st.warning("⚠️ Model belum tersedia. Jalankan `notebook.ipynb` terlebih dahulu untuk melatih model.")
    else:
        st.info("Masukkan data mahasiswa untuk memprediksi kemungkinan dropout.")
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**📋 Data Pribadi**")
                age = st.number_input("Usia Saat Mendaftar", min_value=17, max_value=70, value=20)
                gender = st.selectbox("Gender", ["Perempuan (0)", "Laki-laki (1)"])
                marital = st.selectbox("Status Pernikahan", [
                    "Single (1)", "Menikah (2)", "Duda/Janda (3)",
                    "Cerai (4)", "Persekutuan (5)", "Berpisah (6)"
                ])
                scholarship = st.selectbox("Pemegang Beasiswa", ["Tidak (0)", "Ya (1)"])
                debtor = st.selectbox("Memiliki Hutang", ["Tidak (0)", "Ya (1)"])
                displaced = st.selectbox("Mahasiswa Pindahan", ["Tidak (0)", "Ya (1)"])
                international = st.selectbox("Mahasiswa Internasional", ["Tidak (0)", "Ya (1)"])
                special_needs = st.selectbox("Kebutuhan Pendidikan Khusus", ["Tidak (0)", "Ya (1)"])

            with col2:
                st.markdown("**📚 Data Akademik**")
                tuition = st.selectbox("Biaya Kuliah Tepat Waktu", ["Tidak (0)", "Ya (1)"])
                admission_grade = st.slider("Nilai Penerimaan", 0.0, 200.0, 127.0, step=0.5)
                prev_qual_grade = st.slider("Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 122.0, step=0.5)
                attendance = st.selectbox("Waktu Kuliah", ["Siang (1)", "Malam (0)"])
                app_mode = st.number_input("Mode Pendaftaran", min_value=1, max_value=57, value=17)
                app_order = st.number_input("Urutan Pilihan", min_value=0, max_value=9, value=1)
                course = st.number_input("Kode Program Studi", min_value=33, max_value=9991, value=9254)

            with col3:
                st.markdown("**📝 Kinerja Semester**")
                sem1_enrolled = st.number_input("SKS Sem-1 Diambil", min_value=0, max_value=26, value=6)
                sem1_approved = st.number_input("SKS Sem-1 Lulus", min_value=0, max_value=26, value=6)
                sem1_grade    = st.slider("Nilai Sem-1", 0.0, 20.0, 13.0, step=0.1)
                sem1_eval     = st.number_input("Evaluasi Sem-1", min_value=0, max_value=45, value=6)
                sem2_enrolled = st.number_input("SKS Sem-2 Diambil", min_value=0, max_value=23, value=6)
                sem2_approved = st.number_input("SKS Sem-2 Lulus", min_value=0, max_value=20, value=6)
                sem2_grade    = st.slider("Nilai Sem-2", 0.0, 20.0, 13.0, step=0.1)
                sem2_eval     = st.number_input("Evaluasi Sem-2", min_value=0, max_value=33, value=6)
                unemp_rate    = st.number_input("Tingkat Pengangguran (%)", value=13.9, step=0.1)
                inflation     = st.number_input("Tingkat Inflasi (%)", value=-0.3, step=0.1)
                gdp           = st.number_input("GDP", value=0.79, step=0.01)

            submitted = st.form_submit_button("🔮 Prediksi Sekarang")

        if submitted:
            gender_val    = int(gender.split("(")[1].replace(")", ""))
            marital_val   = int(marital.split("(")[1].replace(")", ""))
            scholar_val   = int(scholarship.split("(")[1].replace(")", ""))
            debtor_val    = int(debtor.split("(")[1].replace(")", ""))
            displaced_val = int(displaced.split("(")[1].replace(")", ""))
            intl_val      = int(international.split("(")[1].replace(")", ""))
            special_val   = int(special_needs.split("(")[1].replace(")", ""))
            tuition_val   = int(tuition.split("(")[1].replace(")", ""))
            attend_val    = int(attendance.split("(")[1].replace(")", ""))

            input_dict = {
                'Marital_status': marital_val, 'Application_mode': app_mode,
                'Application_order': app_order, 'Course': course,
                'Daytime_evening_attendance': attend_val, 'Previous_qualification': 1,
                'Previous_qualification_grade': prev_qual_grade, 'Nacionality': 1,
                'Mothers_qualification': 19, 'Fathers_qualification': 12,
                'Mothers_occupation': 5, 'Fathers_occupation': 9,
                'Admission_grade': admission_grade, 'Displaced': displaced_val,
                'Educational_special_needs': special_val, 'Debtor': debtor_val,
                'Tuition_fees_up_to_date': tuition_val, 'Gender': gender_val,
                'Scholarship_holder': scholar_val, 'Age_at_enrollment': age,
                'International': intl_val,
                'Curricular_units_1st_sem_credited': 0, 'Curricular_units_1st_sem_enrolled': sem1_enrolled,
                'Curricular_units_1st_sem_evaluations': sem1_eval, 'Curricular_units_1st_sem_approved': sem1_approved,
                'Curricular_units_1st_sem_grade': sem1_grade, 'Curricular_units_1st_sem_without_evaluations': 0,
                'Curricular_units_2nd_sem_credited': 0, 'Curricular_units_2nd_sem_enrolled': sem2_enrolled,
                'Curricular_units_2nd_sem_evaluations': sem2_eval, 'Curricular_units_2nd_sem_approved': sem2_approved,
                'Curricular_units_2nd_sem_grade': sem2_grade, 'Curricular_units_2nd_sem_without_evaluations': 0,
                'Unemployment_rate': unemp_rate, 'Inflation_rate': inflation, 'GDP': gdp
            }

            input_df = pd.DataFrame([input_dict])[feature_names]
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            pred_label = le.inverse_transform([pred])[0]

            st.divider()
            box_class = {"Dropout": "result-box-dropout", "Graduate": "result-box-graduate",
                         "Enrolled": "result-box-enrolled"}.get(pred_label, "result-box-enrolled")
            emoji = {"Dropout": "🔴", "Graduate": "🟢", "Enrolled": "🔵"}.get(pred_label, "")

            st.markdown(f"""
            <div class="{box_class}">
                <h2>{emoji} Prediksi: {pred_label}</h2>
                <p style="font-size:1rem; color:#555;">Berdasarkan data yang dimasukkan</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Probabilitas per Kelas")
            prob_df = pd.DataFrame({"Status": le.classes_, "Probabilitas": proba}).sort_values("Probabilitas", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 3))
            colors_bar = ["#E74C3C" if s == "Dropout" else "#2ECC71" if s == "Graduate" else "#3498DB"
                          for s in prob_df["Status"]]
            bars = ax.barh(prob_df["Status"], prob_df["Probabilitas"], color=colors_bar, edgecolor="white")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilitas")
            ax.set_title("Probabilitas Prediksi", fontweight="bold")
            for bar, val in zip(bars, prob_df["Probabilitas"]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{val:.1%}", va="center", fontweight="bold")
            st.pyplot(fig)
            plt.close()

            if pred_label == "Dropout":
                st.error("⚠️ **Rekomendasi:** Mahasiswa ini berisiko dropout. Segera hubungi konselor akademik dan pertimbangkan program intervensi dini.")
            elif pred_label == "Graduate":
                st.success("✅ **Rekomendasi:** Mahasiswa ini memiliki performa baik. Pertahankan dukungan akademik yang ada.")
            else:
                st.info("📌 **Rekomendasi:** Pantau perkembangan mahasiswa ini secara berkala.")

# ── Tab 3 : Analisis Data ─────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📈 Eksplorasi Data Detail")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            feat = st.selectbox("Pilih Fitur Numerik",
                                ["Age_at_enrollment", "Admission_grade", "Previous_qualification_grade",
                                 "Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade",
                                 "Unemployment_rate", "GDP"])
        with col2:
            chart_type = st.selectbox("Jenis Chart", ["Histogram per Status", "Box Plot", "Violin Plot"])

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {"Graduate": "#2ECC71", "Dropout": "#E74C3C", "Enrolled": "#3498DB"}

        if chart_type == "Histogram per Status":
            for status in df["Status"].unique():
                subset = df[df["Status"] == status][feat]
                ax.hist(subset, bins=25, alpha=0.6, label=status, color=colors[status], edgecolor="white")
            ax.set_title(f"Distribusi {feat} per Status", fontweight="bold")
            ax.legend()
        elif chart_type == "Box Plot":
            df.boxplot(column=feat, by="Status", ax=ax, patch_artist=True)
            plt.suptitle("")
            ax.set_title(f"Box Plot {feat}", fontweight="bold")
        else:
            import matplotlib.patches as mpatches
            for i, status in enumerate(df["Status"].unique()):
                vals = df[df["Status"] == status][feat].dropna()
                parts = ax.violinplot(vals, positions=[i], showmedians=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor(colors[status]); pc.set_alpha(0.7)
            ax.set_xticks(range(len(df["Status"].unique())))
            ax.set_xticklabels(df["Status"].unique())
            ax.set_title(f"Violin Plot {feat}", fontweight="bold")

        ax.set_xlabel("Status" if chart_type != "Histogram per Status" else feat)
        ax.set_ylabel("Frekuensi" if chart_type == "Histogram per Status" else feat)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown("**Statistik Deskriptif per Status**")
        st.dataframe(
            df.groupby("Status")[feat].describe().round(3).T,
            use_container_width=True
        )
    else:
        st.error("Dataset tidak tersedia.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.8rem;'>"
    "Jaya Jaya Institut — Student Dropout Prediction System © 2024"
    "</p>",
    unsafe_allow_html=True
)
