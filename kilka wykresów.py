import matplotlib.pyplot as plt









x=[i for i in range(-10,10)]
#y1=[]
#for i in x:
   # y1.append(5*i -2)
#print (y1)

y1=[5*i -2 for i in x]
y2=[-2*i +5 for i in x]
y3=[3*i +3 for i in x]

plt.plot (x,y1, "ro")
plt.plot (x,y2, "b^-")
plt.plot (x, y3, "gs-")
plt.show()