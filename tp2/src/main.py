import numpy as np
from red_neuronal import Red_hebbs

if __name__ == "__main__":
    linea = []
    rta = []
    for i in range(50):
        linea.append(np.array([i,i]))
        rta.append(np.array([i,i]))
    red_hebbs = Red_hebbs(linea,rta)
    red_hebbs.entrenar(0.0001)
    red_hebbs.testear(linea)