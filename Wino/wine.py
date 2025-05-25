import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Wczytanie danych o czerwonym winie
wine_df = pd.read_csv('winequality-red.csv', sep=';') # separator ; w pliku
print("Kolumny:", wine_df.columns)
print(wine_df.head(3))
# 1. Rozkład zmiennej quality
print("\nRozkład jakości (wartości unikalne i ich liczności):")
print(wine_df['quality'].value_counts().sort_index())
# 2. Statystyki opisowe cech fizykochemicznych
print("\nStatystyki opisowe cech:")
print(wine_df.describe())

# 3. Macierz korelacji
corr = wine_df.corr()
print("\nKorelacje z jakością:")
print(corr['quality'].sort_values(ascending=False))
# 4. Wizualizacja korelacji (heatmap)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title('Macierz korelacji cech wina czerwonego')
plt.show()

sns.scatterplot(data=wine_df, x='alcohol', y='quality', alpha=0.5)

plt.title('Jakość wina vs zawartość alkoholu')

plt.show()

wine_df['VA_bin'] = pd.qcut(wine_df['volatile acidity'], 4,

                            labels=['low', 'med_low', 'med_high', 'high'])

sns.boxplot(data=wine_df, x='VA_bin', y='quality')

plt.title('Ocena jakości przy różnych poziomach kwasowości lotnej')

plt.show()
