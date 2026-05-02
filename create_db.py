import pandas as pd
import sqlite3
import os

df = pd.read_csv('data .csv', sep=';')
print(f'Dataset loaded: {df.shape}')

# Simpan ke SQLite
conn = sqlite3.connect('student.db')
df.to_sql('students', conn, if_exists='replace', index=False)

# Verifikasi
count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
print(f'Rows in DB: {count}')

cols = [r[1] for r in conn.execute("PRAGMA table_info(students)").fetchall()]
print(f'Columns: {cols}')

conn.close()
print('student.db created successfully!')
print(f'File size: {os.path.getsize("student.db"):,} bytes')
