import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./personas.csv')

#print(df.head())
#print(df.describe())
#print(df.isna().sum())

df.drop(['Nombre'], axis=1, inplace=True)

X = df.drop(['Peso'], axis=1) # Caracteristicas de entrada
y = df['Peso'] # Variable a predecir

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

#print(predicciones)

print(modelo.predict([[1.70]]))

from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse)

print(mse)
print(rmse)

#¿Funciona bien o no? ¿Por qué cree que es así?
# Se puede observar que el modelo de predicción es medianamente aceptable ya que el rmse es un poco alto pero no tanto,
# esto significaría que según el rmse el modelo no es tan preciso, pero tampoco es tan malo, con una variación de entre 12 a 18
# kg con respecto al peso real,considero que esto pasa en gran parte por la falta de características
# que el modelo pueda juzgar.





