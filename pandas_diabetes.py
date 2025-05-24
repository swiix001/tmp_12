import pandas as pd
import numpy as np

df=pd.read_csv("C:\\Users\\vdi-student\\Desktop\\Analiza_Danych_AI_2\\Zjazd1_12_04_25\\diabetes.csv")
print (df)
print(df.head(3).to_string())
print (f"Kształt danych:{df.shape}")
print(df.describe().T.round(2).to_string())
print ("Ile braków")



for col in ["glucose",  "bloodpressure" , "skinthickness",  "insulin",   "bmi",  "diabetespedigreefunction", "age"]:
   df[col]= df[col].replace(0,np.nan)
   mean_=df[col].mean()
   df[col].replace(np.nan, mean_, inplace=True)


print('\n\n............po czyszczeniu...................\n')
print(df.describe().T.round(2).to_string())
print("Ile braków")
print(df.isna().sum())


df.to_csv("cukrzyca_po_obróbce.csv", index=False, sep=";")



from sklearn.model_selection import train_test_split
x= df.iloc[:,:-1]
y= df.outcome
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train, y_train)
from sklearn.metrics import confusion_matrix
print(pd.DataFrame(confusion_matrix(y_test, model.predict(x_test))))
print(f"Dokładność modelu: {model.score(x_test, y_test)}")
print("Czy klasy zbalansowane?")
print(df.outcome.value_counts())
print("Zmiana danych")
df1=df.query("outcome==0").sample(500)
df2=df.query("outcome==1").sample(500)
df3=pd.concat([df1, df2])
x= df.iloc[:,:-1]
y= df.outcome
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model=LogisticRegression()
model.fit(x_train, y_train)
print(f"Dokładność modelu: {model.score(x_test, y_test)}")
print (pd.DataFrame(confusion_matrix(y_test, model.predict(x_test))))






