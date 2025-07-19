Présenter par : 
MUAMBA MPUTU Jonas 
BITOTA MUKADI Denise 
KONJI KALALA Jean pièrre

PROBLÈME DE RÉGRESSION LINEAIRE EN UTILISANT LE DATASET 'CO2 Emissions_Canada'

# CODE-DU-TP

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


#Chargement des données
data = pd.read_csv('CO2 Emissions_Canada.csv')


#Sélection des variables pertinentes
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
    ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42))
])

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train
X_test

#Entraînement du modèle
model.fit(X_train, y_train)

# Évaluation du modèle 
y_pred = model.predict(X_test)
print("MSE :", mean_squared_error(y_test, y_pred))
print("R² :", r2_score(y_test, y_pred))