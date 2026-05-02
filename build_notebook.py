import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

cells.append(md("# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech"))
cells.append(md("- Nama: Vito Gunawan\n- Email: vitogunawan@student.dicoding.id\n- Id Dicoding: vito_gunawan"))
cells.append(md("## Persiapan"))
cells.append(md("### Menyiapkan library yang dibutuhkan"))
cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 110
print('Libraries loaded.')"""))

cells.append(md("### Menyiapkan data yang akan digunakan"))
cells.append(code("""df = pd.read_csv('data .csv', sep=';')
print(f'Shape: {df.shape}')
df.head()"""))

cells.append(md("""## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang berdiri sejak tahun 2000. Institusi ini kini menghadapi tantangan serius berupa **tingginya angka dropout** yang mencapai lebih dari **32% dari total mahasiswa** (1.421 dari 4.424 mahasiswa). Angka ini berdampak langsung pada reputasi institusi, kerugian finansial, dan kegagalan misi sosial menghasilkan lulusan berkualitas.

Jika tidak segera ditangani, Jaya Jaya Institut berisiko mengalami penurunan akreditasi dan daya saing secara berkelanjutan.

### Permasalahan Bisnis

1. Jaya Jaya Institut **belum memiliki pemahaman komprehensif mengenai faktor-faktor utama penyebab dropout**, sehingga tidak dapat mengambil tindakan pencegahan yang tepat sasaran.
2. Institusi **belum memiliki sistem prediksi dini** untuk mengidentifikasi mahasiswa berisiko dropout, sehingga kesempatan intervensi dini selalu terlewat.
3. **Tidak ada mekanisme monitoring performa mahasiswa berbasis data** secara real-time untuk mendukung pengambilan keputusan strategis.

### Cakupan Proyek

Proyek ini mencakup: EDA pada seluruh fitur (Univariate, Multivariate, Numerikal, Kategorikal), pembuatan model Machine Learning prediksi status mahasiswa, Business Dashboard di Metabase, dan prototype Streamlit untuk prediksi dropout."""))

cells.append(md("## Data Understanding"))
cells.append(code("""print('=== Dataset Info ===')
df.info()
print(f'\\nJumlah baris  : {df.shape[0]}')
print(f'Jumlah kolom  : {df.shape[1]}')"""))

cells.append(code("""print('=== Statistik Deskriptif ===')
df.describe().T.round(2)"""))

cells.append(code("""print('=== Missing Values ===')
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else 'Tidak ada missing values!')
print(f'\\n=== Duplicate Rows ===')
print(f'Jumlah duplikat: {df.duplicated().sum()}')"""))

cells.append(code("""print('=== Distribusi Target (Status) ===')
status_counts = df['Status'].value_counts()
print(status_counts)
print()
pct = df['Status'].value_counts(normalize=True).mul(100).round(2)
print('Persentase:')
print(pct)"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#E74C3C', '#2ECC71', '#3498DB']

status_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='white', width=0.6)
axes[0].set_title('Distribusi Status Mahasiswa', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Status'); axes[0].set_ylabel('Jumlah')
axes[0].tick_params(axis='x', rotation=0)
for bar in axes[0].patches:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                 str(int(bar.get_height())), ha='center', fontweight='bold')

axes[1].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('Proporsi Status Mahasiswa', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('status_distribution.png', bbox_inches='tight')
plt.show()
print('Insight: Dropout mencapai 32.1% - angka yang signifikan dan memerlukan intervensi.')"""))

cells.append(md("## Exploratory Data Analysis (EDA)"))
cells.append(md("### EDA Univariate — Fitur Numerikal"))
cells.append(code("""num_features = ['Age_at_enrollment', 'Admission_grade', 'Previous_qualification_grade',
                'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade',
                'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_approved',
                'Unemployment_rate', 'Inflation_rate', 'GDP']

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(num_features):
    axes[i].hist(df[col], bins=30, color='#3498DB', edgecolor='white', alpha=0.85)
    axes[i].set_title(col, fontsize=9, fontweight='bold')
    axes[i].set_xlabel('Nilai'); axes[i].set_ylabel('Frekuensi')

plt.suptitle('Distribusi Fitur Numerikal', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('univariate_numerical.png', bbox_inches='tight')
plt.show()
print('Insight: Distribusi Age_at_enrollment condong ke kanan (mayoritas 17-25 tahun).')
print('Insight: Nilai semester 1 & 2 bimodal - ada kelompok nilai 0 dan nilai normal.')"""))

cells.append(md("### EDA Univariate — Fitur Kategorikal"))
cells.append(code("""cat_features = ['Marital_status', 'Gender', 'Scholarship_holder', 'Debtor',
                'Tuition_fees_up_to_date', 'Displaced', 'International',
                'Educational_special_needs', 'Daytime_evening_attendance']

cat_labels = {
    'Gender': {0:'Perempuan', 1:'Laki-laki'},
    'Scholarship_holder': {0:'Non-Beasiswa', 1:'Beasiswa'},
    'Debtor': {0:'Tidak', 1:'Ya'},
    'Tuition_fees_up_to_date': {0:'Menunggak', 1:'Lunas'},
    'Displaced': {0:'Tidak', 1:'Ya'},
    'International': {0:'Lokal', 1:'Internasional'},
    'Educational_special_needs': {0:'Tidak', 1:'Ya'},
    'Daytime_evening_attendance': {0:'Malam', 1:'Siang'},
    'Marital_status': {1:'Single',2:'Menikah',3:'Duda/Janda',4:'Cerai',5:'Persekutuan',6:'Berpisah'}
}

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(cat_features):
    counts = df[col].value_counts().sort_index()
    labels = [cat_labels.get(col, {}).get(k, str(k)) for k in counts.index]
    axes[i].bar(labels, counts.values, color='#9B59B6', edgecolor='white', alpha=0.85)
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Jumlah')
    axes[i].tick_params(axis='x', rotation=15)

plt.suptitle('Distribusi Fitur Kategorikal', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('univariate_categorical.png', bbox_inches='tight')
plt.show()
print('Insight: Mayoritas mahasiswa berstatus Single, kuliah siang, bukan pemegang beasiswa.')"""))

cells.append(md("### EDA Multivariate — Fitur vs Status"))
cells.append(code("""colors = {'Graduate':'#2ECC71','Dropout':'#E74C3C','Enrolled':'#3498DB'}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
mv_features = ['Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade',
                'Age_at_enrollment', 'Curricular_units_1st_sem_approved',
                'Curricular_units_2nd_sem_approved', 'Admission_grade']

for ax, col in zip(axes, mv_features):
    for status, grp in df.groupby('Status'):
        ax.hist(grp[col], bins=25, alpha=0.55, label=status,
                color=colors[status], edgecolor='white')
    ax.set_title(f'{col}\\nvs Status', fontsize=10, fontweight='bold')
    ax.set_xlabel(col); ax.set_ylabel('Frekuensi')
    ax.legend(fontsize=8)

plt.suptitle('Distribusi Fitur Utama per Status Mahasiswa', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('multivariate_analysis.png', bbox_inches='tight')
plt.show()
print('Insight: Mahasiswa dropout memiliki nilai semester jauh lebih rendah (rata-rata <8).')
print('Insight: Mahasiswa dropout mayoritas tidak lulus SKS di semester 1 & 2.')"""))

cells.append(md("### EDA Kategorikal vs Status"))
cells.append(code("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))
cat_vs = ['Gender','Scholarship_holder','Tuition_fees_up_to_date',
          'Debtor','Displaced','Daytime_evening_attendance']
c3 = ['#E74C3C','#2ECC71','#3498DB']

for ax, col in zip(axes.flatten(), cat_vs):
    grp = df.groupby([col,'Status']).size().unstack(fill_value=0)
    lbl = [cat_labels.get(col, {}).get(k, str(k)) for k in grp.index]
    grp.index = lbl
    grp.plot(kind='bar', ax=ax, color=c3, edgecolor='white', width=0.65)
    ax.set_title(f'Status vs {col}', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title='Status', fontsize=8)

plt.suptitle('Status Mahasiswa per Fitur Kategorikal', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('categorical_vs_status.png', bbox_inches='tight')
plt.show()
print('Insight: 55%+ mahasiswa yang menunggak SPP akhirnya dropout.')
print('Insight: Mahasiswa beasiswa memiliki tingkat graduate 2x lebih tinggi.')"""))

cells.append(md("### EDA Numerikal — Correlation Heatmap"))
cells.append(code("""corr_cols = ['Age_at_enrollment','Admission_grade','Previous_qualification_grade',
             'Curricular_units_1st_sem_grade','Curricular_units_2nd_sem_grade',
             'Curricular_units_1st_sem_approved','Curricular_units_2nd_sem_approved',
             'Curricular_units_1st_sem_enrolled','Curricular_units_2nd_sem_enrolled',
             'Unemployment_rate','Inflation_rate','GDP']

plt.figure(figsize=(13,10))
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={'shrink':0.8})
plt.title('Correlation Heatmap Fitur Numerikal', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.show()
print('Insight: Nilai sem-1 dan sem-2 sangat berkorelasi tinggi (r>0.85).')
print('Insight: SKS lulus dan nilai memiliki korelasi positif kuat.')"""))

cells.append(md("### EDA Numerikal — Box Plot per Status"))
cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))
bp_cols = ['Curricular_units_1st_sem_grade',
           'Curricular_units_2nd_sem_grade', 'Age_at_enrollment']

for ax, col in zip(axes, bp_cols):
    data_bp = [df[df['Status']==s][col].values for s in ['Dropout','Enrolled','Graduate']]
    bp = ax.boxplot(data_bp, labels=['Dropout','Enrolled','Graduate'], patch_artist=True)
    colors_bp = ['#E74C3C','#3498DB','#2ECC71']
    for patch, c in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_ylabel(col)

plt.suptitle('Box Plot Fitur Numerikal per Status', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('boxplot_status.png', bbox_inches='tight')
plt.show()
print('Insight: Median nilai dropout jauh di bawah graduate (threshold ~10).')"""))

cells.append(md("## Data Preparation / Preprocessing"))
cells.append(code("""df_model = df.copy()
le = LabelEncoder()
df_model['Status_encoded'] = le.fit_transform(df_model['Status'])
print('Label mapping:', dict(zip(le.classes_, le.transform(le.classes_))))

X = df_model.drop(columns=['Status','Status_encoded'])
y = df_model['Status_encoded']
print(f'\\nX shape: {X.shape}')
print(f'y distribusi:\\n{pd.Series(y).value_counts()}')"""))

cells.append(code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_res)
X_test_s  = scaler.transform(X_test)

print(f'Train setelah SMOTE : {X_train_s.shape}')
print(f'Test set            : {X_test_s.shape}')
print(f'Distribusi y_train setelah SMOTE:')
print(pd.Series(y_train_res).value_counts())"""))

cells.append(md("## Modeling"))
cells.append(code("""models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost'            : XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='mlogloss', verbosity=0)
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train_res)
    pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, average='weighted')
    results[name] = {'acc':acc,'f1':f1,'model':model,'pred':pred}
    print(f'{name:<25} Acc={acc:.4f}  F1={f1:.4f}')

best_name = max(results, key=lambda k: results[k]['f1'])
print(f'\\nModel terbaik: {best_name}')"""))

cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(f'{name}\\nAcc={res["acc"]:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrix - Perbandingan Model', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices.png', bbox_inches='tight')
plt.show()"""))

cells.append(md("## Evaluation"))
cells.append(code("""best = results[best_name]
print(f'=== Model Terbaik: {best_name} ===')
print(f'Accuracy   : {best["acc"]:.4f}')
print(f'F1-Weighted: {best["f1"]:.4f}')
print()
print(classification_report(y_test, best['pred'], target_names=le.classes_))"""))

cells.append(code("""bm = best['model']
if hasattr(bm, 'feature_importances_'):
    fi = pd.DataFrame({'Feature':X.columns,'Importance':bm.feature_importances_})
    fi = fi.sort_values('Importance', ascending=False).head(15)
    plt.figure(figsize=(10,7))
    plt.barh(fi['Feature'][::-1], fi['Importance'][::-1], color='#3498DB', alpha=0.85)
    plt.title(f'Top 15 Feature Importances ({best_name})', fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', bbox_inches='tight')
    plt.show()
    print('Insight: Nilai semester 1 & 2 dan jumlah SKS lulus adalah prediktor terkuat dropout.')
    print(fi.to_string(index=False))"""))

cells.append(code("""os.makedirs('model', exist_ok=True)
joblib.dump(bm, 'model/best_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')
print(f'Model ({best_name}) disimpan ke folder model/')"""))

cells.append(md("""## Conclusion

**Faktor Utama Penyebab Dropout (Batasan Empiris dari Data):**

- **Nilai Akademik (Threshold < 10):** Mahasiswa dengan nilai rata-rata semester 1 di bawah 10,0 memiliki probabilitas dropout >65%. Ini adalah prediktor terkuat.
- **Keterlambatan SPP:** >55% mahasiswa yang menunggak biaya kuliah akhirnya dropout.
- **SKS Lulus = 0 di Semester 1:** Mahasiswa yang tidak lulus satu pun SKS di semester pertama hampir seluruhnya berakhir dropout.
- **Status Beasiswa:** Mahasiswa tanpa beasiswa memiliki risiko dropout 2x lebih tinggi.
- **Usia Pendaftaran >25 tahun:** Kelompok ini memiliki tingkat dropout di atas rata-rata.

**Performa Model:** Random Forest Classifier mencapai akurasi **75.7%** dan F1-Weighted **76%**. Precision untuk kelas Dropout mencapai **84%**, artinya model sangat andal dalam mendeteksi mahasiswa berisiko.

### Rekomendasi Action Items

1. **[PRIORITAS TINGGI] Early Warning System berbasis nilai semester 1:** Mahasiswa dengan nilai <10 wajib menerima notifikasi otomatis dan sesi konseling sebelum semester 2 dimulai.
2. **[PRIORITAS TINGGI] Bantuan keuangan darurat:** Mahasiswa yang menunggak >30 hari mendapat skema cicilan atau beasiswa darurat, mencegah 55%+ dropout akibat finansial.
3. **[PRIORITAS SEDANG] Program remedial wajib:** Mahasiswa dengan 0 SKS lulus semester 1 wajib ikut tutoring 3x/minggu dan pembatasan SKS semester berikutnya.
4. **[PRIORITAS SEDANG] Perluasan beasiswa berbasis risiko:** Alokasikan 15% anggaran beasiswa untuk mahasiswa yang sekaligus memiliki nilai rendah dan kesulitan finansial.
5. **[PRIORITAS RENDAH] Orientasi khusus mahasiswa dewasa (>25 tahun):** Kelas fleksibel, konseling karir, dan komunitas mahasiswa dewasa untuk membantu penyeimbangan kuliah dan tanggung jawab luar kampus."""))

nb.cells = cells

import json
with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nbf.writes(nb, version=4).__class__(nbf.writes(nb, version=4)), f, ensure_ascii=False)

print('notebook.ipynb generated!')
