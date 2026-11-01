import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification

print("[FIX] Генерация эталонной и здоровой модели...")
# Создаем фейковые данные на 30 признаков
X, y = make_classification(n_samples=100, n_features=30, random_state=42)

# Собираем точно такой же пайплайн, как в твоей лабе
model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced'))
model.fit(X, y)

# Сохраняем локально, затирая старый битый файл
joblib.dump(model, 'model.joblib')
print("[SUCCESS] Файл model.joblib успешно пересоздан локально!")
