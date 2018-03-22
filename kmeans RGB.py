import cv2
import numpy as np


def dist(a, b): # distanza tra i colori 
    return abs(b-a)
def dist2(XX, CC): # distanza tra i colori
    return np.sum(abs(XX-CC), axis=1)

def visualizzaC(ce,le):
    imc=np.zeros((50,400,3), np.uint8)
    r,c=ce.shape
    s=np.sum(le)
    for i in range(r):
        print "Ce: ", ce[i],"L:",le[i]
        raggio=int(2+le[i]*20./s)
        cv2.circle(imc,(10+i*30,30), raggio, tuple(ce[i]) , 6)
        cv2.imshow('centroifi',imc)         
    return

def generaAcasoCentroidi(k,X):
# coordinate (Cx,Cy) dei k centroidi  random, inserite in un unica matrice C
    return np.random.randint(0, 255, size=k )
    
    

def generaAcasoCentroidi2(k,X):
# coordinate (Cx,Cy) dei k centroidi  random, inserite in un unica matrice C
    C = np.random.randint(0, 255, size=k*3)
    
    return np.resize(C,(k,3))
  
 
im=cv2.imread('aa.jpg')
sh=im.shape
righe,colonne=sh[0],sh[1]
l=3
X = np.resize(im,(righe*colonne,l))
k = 3 # Numero di  clusters 
C=generaAcasoCentroidi2(k,X)
 
 
C_old = np.zeros(C.shape)# Contiene i colori dei centroidi precedenti all'updates
clusters = np.zeros(len(X))# il Cluster a cui appartiente ciascun punto
  
 
for n in range(30 ):
    print "frame", n,C
    im2=X.copy()
   
    # Assegna ad ogni punto della immagine il suo claster
    for i in range(len(X)):
        distances = dist2(X[i], C) #calcola la distanza tra il punto i e i k centroidi
        cluster = np.argmin(distances)#  trova il centroide con distanza minima
        clusters[i] = cluster
        im2[i]=C[cluster]
       
         
    
    C_old = C.copy()# copia le coordinate dei centroidi in C_old
    lpoints=np.zeros(k)
    for i in range(k):
        #in points si copiano tutti punti assegnati al centroide i
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        # si calcolano le nuove coordinate dei centroidi come medie dei punti ad essi assegnat
        lpoints[i]=len(points)
        if len(points)>0:C[i] = np.mean(points, axis=0)

    im3=np.resize(im2, ( righe,colonne,3))
    visualizzaC(C,lpoints) 
    cv2.imshow('lena-' ,im3)
    cv2.waitKey(1000)

    
    if 0== np.sum(dist(C, C_old)):break  #se le nuove i  centroidi non #sono cambiati si esce dal ciclo
 
while 1:
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
            break
cv2.destroyAllWindows()

