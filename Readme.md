# Ejercicio 1: Mejorando el desempeño del modelo de regresión lineal

## ¿El resultado fue mejor o peor?

El resultado mejoró un poco, alrededor de un 20%. Aunque no es una gran mejora, es bastante significativa en regresión lineal, donde pequeños ajustes pueden hacer una gran diferencia en la precisión del modelo.

## ¿Por qué crees que es así? ¿Por qué son necesarios los cambios aplicados?

Los cambios en el tratamiento de datos fueron clave para mejorar el modelo. Se identificaron patrones geográficos y la influencia de los ingresos medios en los precios de las casas. Además, agregué tres nuevas columnas basadas en correlaciones entre variables, lo que permitió capturar relaciones más significativas. También se manejaron los valores atípicos limitando los datos entre el percentil 25 y 75, lo que redujo el impacto de valores extremos y mejoró la precisión de la regresión. Aunque el modelo sigue teniendo dificultades con valores extremos, centrarse en datos representativos mejoró las predicciones significativamente.

# Ejercicio 2: Modelo de Árbol de Decisión

Se calculó: mean_square_error, score, y se visualizó la matriz de correlación y el arbol de decisión.

# Ejercicio 3: Caso de evaluación

## 1. ¿Qué tipo de modelo aplicaría?

Usaría un modelo de clasificación, ya que se trata de una decisión cualitativa entre dos opciones: "reparable" o "nuevo dispositivo" (0 o 1).

## 2. ¿Por qué consideras que ese modelo es adecuado?

Logistic Regression es adecuado para clasificación binaria y funciona bien cuando la relación entre las variables es lineal. DecisionTreeClassifier es útil para manejar relaciones no lineales y permite capturar interacciones complejas entre las características.

## 3. ¿De qué manera considera que es diferente a la programación regular, para este tipo de proyectos?

En la programación regular, escribiríamos reglas específicas para cada caso. En machine learning, el modelo aprende de los datos para hacer predicciones sin reglas explícitas predefinidas.

## 4. ¿Por qué cree que los modelos como los vistos en las clases no logran llegar a un 100% de respuestas correctas?

Ningún modelo logra un 100% de precisión debido a la variabilidad de los datos, el ruido, y las relaciones no lineales. Además, los modelos pueden sufrir de sobreajuste o subajuste si no están bien medidos.
