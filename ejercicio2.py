import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
df = pd.read_csv("./housing.csv")

# Preprocesamiento de datos
df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)
df.dropna(inplace=True)

# Selección de columnas y creación de nuevas características
columnas_seleccionadas = [
    "median_house_value",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "households",
    "longitude",
    "latitude",
    "population",
    "median_income",
    "ocean_proximity_<1H OCEAN",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN",
]
df_seleccionado = df[columnas_seleccionadas].copy()
df_seleccionado["bedrooms_ratio"] = (
    df_seleccionado["total_bedrooms"] / df_seleccionado["total_rooms"]
)
df_seleccionado["incomes_ratio"] = (
    df_seleccionado["median_income"] / df_seleccionado["median_house_value"]
)
df_seleccionado["househould_ratio"] = (
    df_seleccionado["population"] / df_seleccionado["households"]
)

# Filtrado de datos
df_seleccionado = df_seleccionado[
    (df_seleccionado["median_income"] <= 14)
    & (df_seleccionado["median_income"] >= 1.2)
    & (df_seleccionado["housing_median_age"] <= 37)
    & (df_seleccionado["housing_median_age"] >= 18)
    & (df_seleccionado["population"] <= 4000)
    & (df_seleccionado["population"] >= 400)
    & (df_seleccionado["households"] <= 1500)
    & (df_seleccionado["households"] >= 100)
]

sb.heatmap(data=df_seleccionado.corr(), annot=True, cmap="YlGnBu")
plt.show()

# División de características y etiquetas
X = df_seleccionado.drop(["median_house_value"], axis=1)
y = df_seleccionado["median_house_value"]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenamiento del modelo de Árbol de decisión
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)

# Predicción y evaluación
pred = model.predict(X_test)
comparativa = {"predicciones": pred, "valor real": y_test}
result = pd.DataFrame(comparativa)
print(result)

# Evaluación de precisión y error cuadrático medio
score = model.score(X_test, y_test)
print(f"Precisión del modelo: {(score * 100):.2f}%")
print(f"Desviación media: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")

# Visualización del árbol de decisión
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, filled=True, max_depth=3, fontsize=10)
plt.show()
