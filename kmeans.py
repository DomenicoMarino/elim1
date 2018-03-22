
import time
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# ina scelta random dei punti da una distribuzione quasi uniforme sul piano
f1 = np.random.randint(0, 200, size=2299)
f2 = np.random.randint(0, 200, size=2299)


# preferibile una distribuzione particolare, da mrglio l'idea del cluster


data = pd.read_csv('xclara.csv')
print(data.shape)
data.head()
f1 = data['V1'].values
f2 = data['V2'].values

X = np.array(list(zip(f1, f2)))


k = 3   # Numero di  clusters 
# coordinate (Cx,Cy) dei k centroidi  random, inserite in un unica matrice C


C_x = np.random.randint(0, np.max(X)-20, size=k)
C_y = np.random.randint(0, np.max(X)-20, size=k)

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)



# Va in  Plotting di punti di coord f1 e f2 con k Centroidi
fig = plt.figure()
fig.suptitle('Metodo del k-means')
ax1 = fig.add_subplot(221)
ax1.set_title('Prima')
ax2 = fig.add_subplot(222)

ax4 = fig.add_subplot(224)
ax3 = fig.add_subplot(223)
ax4.set_title('clusters finale ')
ax1.scatter(f1, f2, c='black', s=1)
ax1.scatter(C_x, C_y, marker='*', s=100, c='y')

def plotta(nn,X,C,clusters):
    k,cb=C.shape
    print k,len(X),len(clusters)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
   
    
    for i in range(k):
        
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        if (nn==-1): ax4.scatter(points[:, 0], points[:, 1], s=10,c=colors[i])
        if (nn==1): ax2.scatter(points[:, 0], points[:, 1], s=10, c=colors[i])
        if (nn==2): ax3.scatter(points[:, 0], points[:, 1], s=10, c=colors[i])
         
    if (nn==-1):ax4.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='#A505AF')
    if (nn==1):  ax2.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='#A505AF')
    if (nn==2):  ax3.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='#A505AF')
fl=0
nn=0

C_old = np.zeros(C.shape)# Contiene i coordinate dei centroidi precedenti all'updates

clusters = np.zeros(len(X))# il Cluster a cui appartiente ciascun punto

error = dist(C, C_old, None)# Error func. - Distanza tra il nuovo e il vecchio  centroidr 

# Loop fino a quando rror diventa  zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C) #calcola la distanza tra il punti i e i k centroidi
        cluster = np.argmin(distances)#  trova il centroide con distanza minima
        clusters[i] = cluster 
    
    C_old = C.copy() # copia le coordinate dei centroidi in C_old
   
    for i in range(k):
        #in points si copiano tutti punti assegnati al centroide i
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        # si calcolano le nuove coordinate dei centroidi come medie dei punti ad essi assegnat
        C[i] = np.mean(points, axis=0)
    if (nn==0 or nn==1): plotta(nn+1,X,C,clusters)
    nn+=1    
    error = dist(C, C_old, None)  #se le nuove coordinate dei centroidi
                                #sono uguali alle precedenti si esce dal ciclo


plotta(-1,X,C,clusters)
print "numero di cicli", nn
plt.show()
