import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data .csv', sep=';')
print('Data loaded:', df.shape)
print('Status:', df['Status'].value_counts().to_dict())

le = LabelEncoder()
y = le.fit_transform(df['Status'])
X = df.drop(columns=['Status'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_res)
X_test_s  = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_s, y_train_res)
pred = model.predict(X_test_s)

acc = accuracy_score(y_test, pred)
f1  = f1_score(y_test, pred, average='weighted')
print(f'Accuracy: {acc:.4f}')
print(f'F1 Weighted: {f1:.4f}')
print(classification_report(y_test, pred, target_names=le.classes_))

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/best_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')
print('Model saved to model/ folder!')
