import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./housing.csv")

# print(df.head())
# print(df.info())
# print(df.describe())

# Identificación de límites y medición de datos

# print(df["housing_median_age"].value_counts())
# print(df["median_house_value"].value_counts())
# print(df["median_house_value"].nlargest(10).unique())
# print(df["median_income"].value_counts())
# print(df["population"].value_counts())
# print(df["latitude"].value_counts())
# print(df["households"].value_counts())

# Transformación de datos
df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)
df.dropna(inplace=True)

# df.info()

# Selección de las columnas deseadas
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

# Crear un nuevo DataFrame con solo las columnas seleccionadas
df_seleccionado = df[columnas_seleccionadas].copy()

# Crear nuevas columnas para mejorar relaciones
df_seleccionado["rooms_ratio"] = (
    df_seleccionado["total_bedrooms"] / df_seleccionado["total_rooms"]
)
df_seleccionado["incomes_ratio"] = (
    df_seleccionado["median_income"] / df_seleccionado["median_house_value"]
)
df_seleccionado["househould_ratio"] = (
    df_seleccionado["population"] / df_seleccionado["households"]
)

# Filtrado de datos para quitar extremos altos y bajos
df_seleccionado = df_seleccionado[
    (df_seleccionado["median_income"] <= 14) & (df_seleccionado["median_income"] >= 1.2)
]

df_seleccionado = df_seleccionado[
    (df_seleccionado["housing_median_age"] <= 37)
    & (df_seleccionado["housing_median_age"] >= 18)
]

df_seleccionado = df_seleccionado[
    (df_seleccionado["population"] <= 4000) & (df_seleccionado["population"] >= 400)
]

df_seleccionado = df_seleccionado[
    (df_seleccionado["households"] <= 1500) & (df_seleccionado["households"] >= 100)
]

# Analisis de datos extremos en graficos

# print(df_seleccionado.describe())

sb.heatmap(data=df_seleccionado.corr(), annot=True, cmap="YlGnBu")
plt.show()

# Grafica la relacion de latitude, longitude con el precio de las casa
# sb.scatterplot(
#     x="longitude",
#     y="latitude",
#     hue="median_house_value",
#     data=df_seleccionado,
#     palette="viridis",
# )
# plt.show()

# sb.regplot(x="median_income", y="median_house_value", data=df_seleccionado)
# plt.show()

# sb.regplot(x="housing_median_age", y="median_house_value", data=df_seleccionado)
# plt.show()

# sb.regplot(x="population", y="median_house_value", data=df_seleccionado)
# plt.show()

# sb.regplot(x="households", y="median_house_value", data=df_seleccionado)
# plt.show()


# División de características y etiquetas
X = df_seleccionado.drop(["median_house_value"], axis=1)
y = df_seleccionado["median_house_value"]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción y evaluación
pred = model.predict(X_test)
comparativa = {"predicciones": pred, "valor real": y_test}
result = pd.DataFrame(comparativa)
print(result)


score = model.score(X_test, y_test)
print(
    f"Precisión del modelo: {(score * 100):.2f}%",
)
print(
    f"Desviación media: {np.sqrt(mean_squared_error(y_test, pred)):.2f}",
)
