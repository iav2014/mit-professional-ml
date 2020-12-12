import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('prices.csv', parse_dates=[0], header=None, index_col=0, squeeze=True,
                 names=['fecha', 'unidades'])
print("df.head():", df)
# Veamos algunas gráficas sobre los datos que tenemos del fondo
print("df.describe():", df.describe())
# Por ejemplo, podemos ver de qué fechas tenemos datos con:
print("df.index.min():", df.index.min())
print("df.index.max():", df.index.max())

# Presumiblemente tenemos los valores liquidativos del fondo diarias de 2017,2018 y 2019 hasta el mes de noviembre (14).
# Y ahora veamos cuantas muestras tenemos de cada año:
print("items de 2017:", len(df['2017']))
print("items de 2018:", len(df['2018']))
print("items de 2018:", len(df['2019']))
#  promedios mensuales:
meses = df.resample('M').mean()
print("meses:", meses)

dias = df.resample('D').mean()
print("dias:", dias)

anios = df.resample('A').mean()
print("años:", anios)
# Y visualicemos esas medias mensuales por año
import matplotlib.pylab as plt

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
plt.plot(anios['2017'].values)
plt.plot(anios['2018'].values)
plt.plot(anios['2019'].values)
plt.show()

plt.plot(meses['2017'].values)
plt.plot(meses['2018'].values)
plt.plot(meses['2019'].values)
plt.show()

plt.plot(dias['2017'].values)
plt.plot(dias['2018'].values)
plt.plot(dias['2019'].values)
plt.show()
# grafico meses de verano: de junio a septiembre
verano2017 = df['2017-06-01':'2017-09-01']
plt.plot(verano2017.values)
verano2018 = df['2018-06-01':'2018-09-01']
plt.plot(verano2018.values)
verano2019 = df['2019-06-01':'2019-09-01']
plt.plot(verano2019.values)
plt.show()

PASOS = 7

# convertimos la serie a un problema de entrenamiento supervisado
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = pd.DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg


# load dataset
values = df.values
# los liquidativos son floats
values = values.astype('float32')
# normalize
scaler = MinMaxScaler(feature_range=(-1, 1))

values = values.reshape(-1, 1)

scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, PASOS, 1)
print(reframed.head())

# split into train and test sets, dividimos el set de entrenamiento para pruebas
# ultimos 30 items para test y validación
values = reframed.values
n_train_days = 315 + 289 - (30 + PASOS)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
#
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

# La arquitectura de la red neuronal será:
# (7 inputs, 1 hidden 7 neuronas, 1 salida)
# Entrada 7 inputs, como dijimos antes

# Como función de activación tangente hiperbolica ya que la salida son valores entre  -1 y 1.
# Utilizaremos como optimizador Adam y métrica de pérdida (loss) Mean Absolute Error (lo recomiendan en otros ejemplos vistos)
# Como la predicción será un valor continuo y no discreto,
# para calcular el Acuracy utilizaremos Mean Squared Error y para saber si mejora con el entrenamiento se debería ir reduciendo con las EPOCHS.

def crear_modeloFF():
  model = Sequential()
  model.add(Dense(PASOS, input_shape=(1, PASOS), activation='tanh'))
  model.add(Flatten())
  model.add(Dense(1, activation='tanh'))
  model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=["binary_accuracy"]) #mse
  model.summary()
  return model

EPOCHS = 70

model = crear_modeloFF()

history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=PASOS)

# Visualizamos al conjunto de validación (recordemos que eran 30 días)

results = model.predict(x_val)
plt.scatter(range(len(y_val)), y_val, c='g')
plt.scatter(range(len(results)), results, c='r')
plt.title('validate')
plt.show()

# preciccion de la ultima 2 semana de noviembre en base a la primera quincena de novielbre de 2019
# nota que la ultima semana de noviembre de 2019 no figura en el dataset
ultimosDias = df['2019-11-01':'2019-11-14']
print("valor liquidativo del fondo primeras dos semanas de noviembre 2019")
print(ultimosDias)

values = ultimosDias.values
values = values.astype('float32')
# normalize features
values = values.reshape(-1, 1)  # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.drop(reframed.columns[[7]], axis=1, inplace=True)
# pintamos solo los ultimos 7 dias de la segunda de noviembre, todas las filas
print(reframed.head(7))
# escogemos solo la segunda de noviembre su ultima columna (7)
values = reframed.values
print("values", values);
x_test = values[6:, :]
print("x_test:", x_test)
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_test)


# Ahora crearemos una función para ir «rellenando» el desplazamiento que hacemos por cada predicción.
#  Esto es porque queremos predecir los 7 primeros días siguientes (del 14 + 7 dias de noviembre).

def agregarNuevoValor(x_test, nuevoValor):
  for i in range(x_test.shape[2] - 1):
    x_test[0][0][i] = x_test[0][0][i + 1]
  x_test[0][0][x_test.shape[2] - 1] = nuevoValor
  return x_test


results = []
for i in range(7):
  parcial = model.predict(x_test)
  results.append(parcial[0])
  print(x_test)
  x_test = agregarNuevoValor(x_test, parcial[0])

# convertimos el resultado en valores escalares ya que estan entre -1 y +1
adimen = [x for x in results]
inverted = scaler.inverse_transform(adimen)
print(inverted)

prediccion1SemanaDiciembre = pd.DataFrame(inverted)
prediccion1SemanaDiciembre.columns = ['pronostico']
prediccion1SemanaDiciembre.plot()
prediccion1SemanaDiciembre.to_csv('pronostico.csv')

plt.plot(prediccion1SemanaDiciembre.values)
plt.show()
