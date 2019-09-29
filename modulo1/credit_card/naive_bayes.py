# GAUSIAN NAIVE BAYES
# Nacho

# neccesary imports
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# load data from dataset (csv)
# cargamos los datos de entrada

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB  # nuestro algotirmo estadistico
from sklearn.feature_selection import SelectKBest

data = pd.read_csv("./Credit_Risk_Train_data.csv")
# eliminamos la info que creemos que es redundante, de esta forma
# al haber menos columnas para la prediccion aumentamos el accuracy pero tenemos mas sesgo

data.fillna(0, inplace=True);
reduced = data.drop(['Education', 'Self_Employed', 'Married', 'Property_Area'], axis=1)
reduced.describe()

# visualizamos nuestros datos, media, mediana, std,etc
data = reduced
print("data.shape():", data.shape)
print("data.head():", data.head(17))
print("data.describe():", data.describe())

# visualizamos cuantos items tenemos de uno u otro valor (Y/N)
print(data.groupby('Loan_Status').size())

# preparamos los datos
# a la funcion debemos pasarle valores numericos, para ello, hemos reemplazado
# el valor de algunos campos por un 0 o un 1

del data['Loan_ID']
data.Gender = data.Gender.replace({"Male": 0, "Female": 1})
data.Dependents = data.Dependents.replace({"3+": 3, "0": 0, "1": 1, "2": 2})
# estos datos ya no existen, pero los dejamos comentados para probar
# data.Married=data.Married.replace({"No": 0, "Yes": 1})
# data.Education=data.Education.replace({"Not Graduate":0 , "Graduate": 1})
# data.Self_Employed=data.Self_Employed.replace({"No":0,"Yes":1})
# data.Property_Area=data.Property_Area.replace({"Rural":0,"Urban":1,"Semiurban":2})
data.Loan_Status = data.Loan_Status.replace({"Y": 1, "N": 0})

# data['total amount']=data['LoanAmount']*1000

data.drop('Loan_Status', axis=1).hist()
# visualizamos los datos
# graficas de ingresos por numero de muestras (cliente y participe)
# grafica por genero (sobre total de items)
# grafica de historial de credito por numero de items
# grafico de cantidades a pagar por numero de items
# grafico de numero de descendientes por numero de items,
plt.show();
# por las graficas de los datos deducimos que la mayoria de los clientes que piden credito
# lo hacen con ingresos entre 600 y 10.000, sin hijos y la media del importe de 350 y genero masculino


y = data['Loan_Status']

X = data.drop(['Loan_Status'], axis=1)
y = data['Loan_Status']

# seleccion del algoritmo, escojemos todas las columnas para alimentar el modelo
best = SelectKBest(k=7)
X_new = best.fit_transform(X, y)
X_new.shape
selected = best.get_support(indices=True)
print(X.columns[selected])

used_features = X.columns[selected]

# se obtiene la matriz de correlaccion

colormap = plt.cm.viridis
plt.figure(figsize=(12, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sb.heatmap(data[used_features].astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap,
           linecolor='white', annot=True)
plt.show()

# reservamos el 20% de nuestro set de datos para realizar las pruebas
# en dataset en training y test datasets
X_train, X_test = train_test_split(data, test_size=0.2, random_state=6)
y_train = X_train["Loan_Status"]
y_test = X_test["Loan_Status"]

# Instantiate the classifier
gnb = GaussianNB()
# Train classifier
gnb.fit(
    X_train[used_features].values,
    y_train
)
y_pred = gnb.predict(X_test[used_features])

print('Precisión en el set de Entrenamiento: {:.2f}'
      .format(gnb.score(X_train[used_features], y_train)))
print('Precisión en el set de Test: {:.2f}'
      .format(gnb.score(X_test[used_features], y_test)))

pd.set_option('display.max_columns', None)
print("data.head():", data.head(17))
# efectuamos una prediccion
b = data[0:10]
bb = b.drop(['Loan_Status'], axis=1)
print(gnb.predict(bb))
# data.drop(['Loan_Status'],1).hist()
# plt.show()

# sb.pairplot(data.dropna(), hue='Loan_Status',size=2,vars=["ApplicantIncome","CoapplicantIncome"],kind='reg')
#
# del data['Loan_ID']
# data.Gender=data.Gender.replace({"Male": 0, "Female": 1})
# data.Dependents=data.Dependents.replace({"3+": 3,"0":0,"1":1,"2":2})
# data.Married=data.Married.replace({"No": 0, "Yes": 1})
# data.Education=data.Education.replace({"Not Graduate":0 , "Graduate": 1})
# data.Self_Employed=data.Self_Employed.replace({"No":0,"Yes":1,"":0})
#

# data.Property_Area=data.Property_Area.replace({"Rural":0,"Urban":1,"Semiurban":2})
# data.Loan_Status=data.Loan_Status.replace({"Y":1,"N":0})


# data.fillna(0, inplace=True);
# pd.set_option('display.max_columns', None)
# print(data.head(5))


# X = np.array(data.drop(['Loan_Status'],1))
# y = np.array(data['Loan_Status'])
# X.shape

# aqui probamos logistic regresion
# model = linear_model.LogisticRegression()
# model.fit(X,y)


# predictions = model.predict(X)
# print(model.score(X,y))
# print(predictions)[0:1]
# P=np.array([0,0.0,0,1,0,5849,0.0,0.0,360.0,1.0,0]);
# print(X[0:2])
# print(model.predict(X[0:50]))
