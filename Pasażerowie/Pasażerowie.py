import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

flights = pd.read_csv('AirPassengers.csv')
# Ewentualna konwersja daty
flights['Date'] = pd.to_datetime(flights['Month'], format='%Y-%m')
flights.set_index('Date', inplace=True)
print(flights.shape)
print(flights.head(5))
print(flights.info())
print("\nStatystyki pasażerów:")
print(flights['#Passengers'].describe())
# 1. Wykres liniowy pasażerów w czasie
plt.plot(flights['#Passengers'])
plt.title('Liczba pasażerów linii lotniczych (1949-1960)')
plt.xlabel('Miesiąc (kolejno)')
plt.ylabel('Liczba pasażerów (tys.)')
plt.show()

import seaborn as sns

flights['Year'] = flights.index.year
flights['Month_name'] = flights.index.month_name()

# Pivot: rok jako wiersze, miesiące jako kolumny
pivot_data = flights.pivot(index='Year', columns='Month_name', values='#Passengers')

months_order = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]
pivot_data = pivot_data[months_order]

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_data, cmap="YlOrBr", linewidths=0.5)
plt.title('Liczba pasażerów w poszczególnych miesiącach (1949–1960)')
plt.xlabel('Miesiąc')
plt.ylabel('Rok')
plt.tight_layout()
plt.show()