import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./personas_conEdad.csv')

#print(df.head())
#print(df.describe())
#print(df.isna().sum())

df.drop(['Nombre'], axis=1, inplace=True)
print(df)

X = df.drop(['Peso'], axis=1) # Caracteristicas de entrada
y = df['Peso'] # Variable a predecir

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

modelo = LinearRegression()

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)


#print(predicciones)

print(modelo.predict([[1.70, 19]]))

from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse)

print(mse)
print(rmse)

#¿Funciona mejor el modelo? ¿Por qué cree que es así?
# Se puede observar que el modelo de prediccion de peso mejora un poco, ya que el rmse es menor en comparacion ya que va de 10 a 16
# al modelo anterior, considero que esto pasa en gran parte por agregar una nueva caracteristica que el modelo pueda juzgar
# para mejorar aun mas su prediccion habria que agregar mas caracteristicas


