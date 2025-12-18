import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Загрузка данных
training_set = pd.read_csv('train.csv')

# ==================== ПОДГОТОВКА ДАННЫХ ====================

# Целевая переменная
labels = training_set["Transported"]
l_enc = LabelEncoder()
prep_labels = l_enc.fit_transform(labels)  # True->1, False->0

# Удаляем ненужные колонки
training_set = training_set.drop(["Transported", "Name"], axis=1)

# ==================== FEATURE ENGINEERING ====================

# 1. Извлекаем номер группы из PassengerId
training_set["Group"] = training_set["PassengerId"].str.split("_").str[0]

# 2. Разбиваем Cabin на составляющие
training_set["Deck"] = training_set["Cabin"].str.split("/").str[0]
training_set["Side"] = training_set["Cabin"].str.split("/").str[2]

# 3. Создаем признак "Общие расходы"
training_set["TotalSpent"] = (
    training_set["RoomService"] + 
    training_set["FoodCourt"] + 
    training_set["ShoppingMall"] + 
    training_set["Spa"] + 
    training_set["VRDeck"]
)

# 4. Создаем признак "Есть ли расходы вообще"
training_set["HasSpent"] = (training_set["TotalSpent"] > 0).astype(int)

# 5. Размер группы (сколько людей с одинаковым Group)
group_sizes = training_set["Group"].value_counts()
training_set["GroupSize"] = training_set["Group"].map(group_sizes)

# Удаляем исходные колонки
training_set = training_set.drop(["Cabin", "PassengerId"], axis=1)

# ==================== ОПРЕДЕЛЕНИЕ ПРИЗНАКОВ ====================

# Числовые признаки
num_columns = [
    "Age", 
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "TotalSpent",
    "GroupSize"
]

# Категориальные признаки
cat_columns = [
    "HomePlanet", 
    "CryoSleep", 
    "Destination", 
    "VIP",
    "Deck", 
    "Side",
    "Group",
    "HasSpent"  # Категориальный, т.к. 0/1
]

# ==================== ПАЙПЛАЙНЫ ====================

# Пайплайн для числовых признаков
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Пайплайн для категориальных признаков
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Объединяющий трансформер
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_columns),
    ("cat", cat_pipeline, cat_columns)
])

# ==================== ПОДГОТОВКА ДАННЫХ ====================

# Преобразуем все данные
train_set_prepared = full_pipeline.fit_transform(training_set)

# ==================== МОДЕЛИ ====================


test_data = train_set_prepared[:5]
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(train_set_prepared, prep_labels)

rf_predictions = rf.predict(test_data)
print(f"Первые 5 предсказаний: {rf_predictions}")
print(f"Реальные значения:      {prep_labels[:5]}")

rf_accuracy = accuracy_score(prep_labels, rf.predict(train_set_prepared))
print(f"\nAccuracy Random Forest: {rf_accuracy:.2%}")




