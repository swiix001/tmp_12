import pandas as pd
df = pd.read_csv("train Titanic.csv")

# zakładamy, że nazwa pliku to titanic_train.csv

# 1. Podstawowe informacje o zbiorze
print("Liczba wierszy i kolumn:", df.shape)  # rozmiar danych
print(df.columns)  # nazwy kolumn
print(df.head(5))  # podgląd pierwszych 5 rekordów

# Sprawdzenie braków danych w poszczególnych kolumnach
print("\nBrakidanych w kolumnach:")
print(df.isnull().sum())  # liczba braków (NaN) w każdej kolumnie

# 2. Statystyki opisowe dla zmiennych numerycznych
print("\nStatystykiopisowe zmiennych numerycznych:")
print(df.describe())

# Sprawdźmy dane numeryczne
num_cols = df.select_dtypes(include='number').columns
print("Kolumny numeryczne:", list(num_cols))

# Miary położenia i rozproszenia
print("\n Statystyki opisowe:")
print(df[num_cols].describe())

# Miary położenia i kształtu dla kolumny 'Age'
age = df["Age"].dropna()

print(f"\n Średnia wieku: {age.mean():.2f}")
print(f"Medianawieku: {age.median():.2f}")
print(f"Kwartyl1 (Q1): {age.quantile(0.25):.2f}")
print(f"Kwartyl3 (Q3): {age.quantile(0.75):.2f}")
print(f"Rozstępmiędzykwartylowy(IQR): {age.quantile(0.75) - age.quantile(0.25):.2f}")

from scipy.stats import skew, kurtosis

# Miary kształtu
print(f"\n Skośność (skewness): {skew(age):.2f}")
print(f"Kurtoza(kurtosis): {kurtosis(age):.2f}")

# Histogram wieku z gęstością
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(8, 4))
sns.histplot(age,kde=True,bins=30,color='skyblue')
plt.title("Rozkład wieku pasażerów")
plt.xlabel("Wiek")
plt.ylabel("Liczba pasażerów")
plt.axvline(age.mean(),color='red',linestyle='--',label='Średnia')
plt.axvline(age.median(),color='green',linestyle='--',label='Mediana')
plt.legend()
plt.tight_layout()
plt.show()

# Wybrana zmienna numeryczna
kolumna = 'Age'

# Histogram z KDE
plt.figure(figsize=(10, 6))
sns.histplot(df[kolumna].dropna(), kde=True, bins=30)
plt.title(f'Rozkład zmiennej: {kolumna}')
plt.xlabel('Wartość')
plt.ylabel('Częstość')
plt.show()

# Wykres pudełkowy
plt.figure(figsize=(8, 6))
sns.boxplot(y=df[kolumna])
plt.title(f'Wykres pudełkowy: {kolumna}')
plt.ylabel('Wartość')
plt.show()

# 3. Analiza przeżywalności w grupach kategorycznych
# Tabela częstości: płeć vs przeżycie
survival_by_sex= pd.crosstab(df['Sex'],df['Survived'])
print("\nPrzeżyciewzględem płci:")
print(survival_by_sex)
# Obliczenie procentu przeżycia dla płci
survival_rate_by_sex= survival_by_sex.div(survival_by_sex.sum(axis=1),axis=0) * 100
print("\n% przeżycia wg płci:")
print(survival_rate_by_sex)
# Tabela częstości: klasa vs przeżycie
survival_by_class= pd.crosstab(df['Pclass'],df['Survived'])
print("\nPrzeżyciewzględem klasy kabiny:")
print(survival_by_class)
# % przeżycia dla każdej klasy
survival_rate_by_class= survival_by_class.div(survival_by_class.sum(axis=1),axis=0) * 100
print("\n% przeżycia wg klasy:")
print(survival_rate_by_class)

# Analiza przeżywalności według klasy pasażerskiej na podstawie wczytanego pliku
survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100

# Wykres
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=survival_by_class.index, y=survival_by_class.values)
plt.title('Procent ocalałych według klasy pasażerskiej')
plt.xlabel('Klasa pasażerska')
plt.ylabel('Procent ocalałych')

# Dodanie etykiet z wartościami na słupkach
for i, v in enumerate(survival_by_class.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center')

plt.tight_layout()
plt.show()

# Analiza przeżywalności według płci

survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100

print(survival_by_sex)

plt.figure(figsize=(8, 6))

ax = sns.barplot(x=survival_by_sex.index, y=survival_by_sex.values)

plt.title('Procent ocalałych według płci')

plt.xlabel('Płeć')

plt.ylabel('Procent ocalałych')

# Dodanie etykiet z wartościami na słupkach

for i, v in enumerate(survival_by_sex.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center')

plt.tight_layout()

plt.show()

# Analiza wieku pasażerów

plt.figure(figsize=(12, 6))

sns.histplot(data=df, x='Age', hue='Survived',

             kde=True, element='step', common_norm=False,

             palette=['red', 'green'])

plt.title('Rozkład wieku pasażerów według przeżycia')

plt.xlabel('Wiek')

plt.ylabel('Liczba pasażerów')

plt.legend(title='Przeżył', labels=['Nie', 'Tak'])

plt.tight_layout()

plt.show()

# Statystyki wieku

print(df['Age'].describe())

# Brakujące wartości wieku

print(f"Liczba brakujących wartości wieku: {df['Age'].isnull().sum()}")

print(f"Procent brakujących wartości wieku: {df['Age'].isnull().mean() * 100:.1f}%")

plt.figure(figsize=(12, 6))
sns.boxplot(x='Sex', y='Age', data=df)
plt.title('Wiek pasażerów wg płci (z punktami)')
plt.xlabel('Płeć')
plt.ylabel('Wiek')
plt.grid(True, alpha=0.3)
plt.show()