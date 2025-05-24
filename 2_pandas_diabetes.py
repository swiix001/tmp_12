import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\vdi-student\\Desktop\\Analiza_Danych_AI_2\\Zjazd1_12_04_25\\diabetes.csv')
print(df)
print(df.head(3).to_string())
print(f'Kształt DANYCH: {df.shape}')
print(df.describe().T.round(2).to_string())
print('Ile brakow')
print(df.isna().sum())

# tam, gdzie zero albo brak wartości - dac średnią (bez zer)
for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin',
       'bmi', 'diabetespedigreefunction', 'age']:
    df[col] = df[col].replace(0, np.nan)
    mean_ = df[col].mean()
#    df[col].replace(np.nan, mean_, inplace=True)  # przestanie działać
    df[col] = df[col].replace(np.nan, mean_)

print('\n\n..............po czyszczeniu..............\n')
print(df.describe().T.round(2).to_string())
print('Ile brakow')
print(df.isna().sum())

df.to_csv('cukrzyca_po_obrobce.csv', index=False, sep=';')


# teraz machine learning
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1]
y = df.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
print(f'Dokładność modelu: {model.score(X_test, y_test)}')
print(df.outcome.value_counts())
