# Logictic regresion,sklearn
# Nacho Ariza MIT professional

# neccesary imports
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


data = pd.read_csv("./Credit_Risk_Train_data.csv")

# transformamos los datos para que el algoritmo pueda trabajar ocn ellos

data.fillna(0, inplace=True);
dataframe = data.drop(['Education', 'Self_Employed', 'Married', 'Property_Area'], axis=1)

print(dataframe);
print(dataframe.describe())
del dataframe['Loan_ID']

dataframe.Gender = dataframe.Gender.replace({"Male": 0, "Female": 1})
dataframe.Dependents = dataframe.Dependents.replace({"3+": 3, "0": 0, "1": 1, "2": 2})
dataframe.Loan_Status =dataframe.Loan_Status.replace({"N": 0, "Y": 1,"":0})

dataframe.fillna(0, inplace=True);

# graficas sobre total de registros / campos de estudio
print(dataframe.groupby('Loan_Status').size())
dataframe.drop(['Loan_Status'],1).hist()
plt.show()
#sb.pairplot(dataframe.dropna(), hue='Loan_Status',height=4,vars=["Gender", "ApplicantIncome","LoanAmount","Loan_Amount_Term"],kind='reg')

X = np.array(dataframe.drop(['Loan_Status'],1))
y = np.array(dataframe['Loan_Status'])
X.shape


# set de train (80% y de test 20%)
X_train, X_test = train_test_split(dataframe, test_size=0.2, random_state=6)
y_train = X_train["Loan_Status"]
y_test = X_test["Loan_Status"]


model = linear_model.LogisticRegression()
model.fit(X,y)
print('model score',model.score(X,y))
del X_test['Loan_Status']


# este es el set de datos que usaremos para de TEST
print(X_test)
predictions = model.predict(X_test)
print(predictions)