Présenter par : 
MUAMBA MPUTU Jonas 
BITOTA MUKADI Denise 
KONJI KALALA Jean-pièrre

# CODE-DU-TP

PROBLÈME DE RÉGRESSION LINEAIRE
DATASET:'CO2 Emissions_Canada'

# importations 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Insertion du fichier dans l'environnement google colab 
from google.colab import files
uploaded = files.upload()
print(uploaded)

# Chargement des données
data = pd.read_csv('CO2 Emissions_Canada.csv')

# Sélection des variables pertinentes
features = ['Engine Size(L)', 'Cylinders', 'Fuel Type', 'Transmission', 'Vehicle Class']
target = 'CO2 Emissions(g/km)'
X = data[features]
y = data[target]

# Prétraitement (encodage des variables catégorielles)
categorical = ['Fuel Type', 'Transmission', 'Vehicle Class']
numerical = ['Engine Size(L)', 'Cylinders']

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

# Modèle ensembliste : Gradient Boosting Regressor
model = Pipeline([
    ('preproc', preprocessor),
    ('gbr', GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, random_state=42))
])

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train
X_test

# Entraînement du modèle
model.fit(X_train, y_train)

# Évaluation du modèle 
# (1) Prédiction
y_pred = model.predict(X_test)

# (2) Calcul des métriques de régression
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# (3) Affichage des résultats
print("\nRapport de régression après test:")
print(f"MAE (Erreur Absolue Moyenne): {mae:.4f}")
print(f"MSE (Erreur Quadratique Moyenne): {mse:.4f}")
print(f"RMSE (Racine de l'Erreur Quadratique Moyenne): {rmse:.4f}")
print(f"R² (Coefficient de détermination): {r2:.4f}")

# --------------FIN--------------



PROBLEME DE CLASSIFICATION
DATASET : 'Iris'

# importations
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Chargement
data = load_iris()
X = data.data
y = data.target

# Encodage catégoriel via Keras
y_encoded = to_categorical(y)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Construction du modèle Deep Learning
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.1)

# Évaluation du modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# --------------FIN--------------
