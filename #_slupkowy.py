import matplotlib.pyplot as plt
import random

names=["Kamil", "Kasia", "Bartosz", "Maja", "piesek"]
points= [12,4,17,31,5]
points=[random.randint(a:3,b:15) for name in names]

plt.bar(names, points, color=["green", "blue", "yellow"])
plt.xticks(names)
plt.yticks(points)
plt.show()