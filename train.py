import os
import io
import joblib
import boto3
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, recall_score

load_dotenv()

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("MINIO_ENDPOINT_URL")
)

print("[INFO] Скачивание данных из локального бакета S3...")
response = s3_client.get_object(Bucket=BUCKET_NAME, Key="creditcard.csv")
df = pd.read_csv(io.BytesIO(response["Body"].read()))

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[INFO] Обучение модели на Mac...")
# 1. Добавь эти импорты в самый верх твоего файла train.py
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 2. Замени строчку создания модели (LogisticRegression) на эту конструкцию:
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
)


model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
recall_val = recall_score(y_test, y_pred)

print("\n=== РЕЗУЛЬТАТЫ МЕТРИК ===")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Recall: {recall_val:.4f}\n")

model_buffer = io.BytesIO()
joblib.dump(model, model_buffer)
model_buffer.seek(0)

print(f"[INFO] Отправка обученной модели model.joblib в бакет...")
s3_client.put_object(Bucket=BUCKET_NAME, Key="model.joblib", Body=model_buffer.getvalue())
print("[SUCCESS] Готово!")
