import numpy as np
import matplotlib.pyplot as plt

class Red_hebbs:
    def __init__(self, inputs, targets):
        self.cantidad_neuronas_entrada = np.shape(inputs)[1]
        self.cantidad_neuronas_salida = np.shape(targets)[1]
        self.cantidad_instancias_dataset = np.shape(inputs)[0]

        self.inputs_entrenamiento = inputs
        self.targets_entrenamiento = targets

        self.weights = np.random.normal(size=(self.cantidad_neuronas_entrada, self.cantidad_neuronas_salida),scale= 1/np.sqrt(self.cantidad_neuronas_entrada))

        np.seterr(all='raise')

    def entrenar(self,learning_rate):
        for i in range(1000):
            for instancia in self.inputs_entrenamiento:
                y = np.dot(instancia, self.weights)
                x_raya = np.zeros(self.cantidad_neuronas_entrada)
                delta_weights = np.zeros((self.cantidad_neuronas_entrada,self.cantidad_neuronas_salida))
                for j in range(self.cantidad_neuronas_salida):
                    for i in range(self.cantidad_neuronas_entrada):
                        for k in range(j+1):
                            x_raya[i] += y[k] * self.weights[i][k]
                        delta_weights[i,j] = learning_rate * (instancia[i] - x_raya[i])*y[j]
                self.weights = self.weights + delta_weights

    def testear(self,inputs):
        rangox = 1
        rangoy = 1
        resultado = np.dot(inputs, self.weights)
        print resultado
        plt.scatter(resultado.T[0],resultado.T[1],resultado.T[1])
        #plt.axis([np.amin(resultado[1])-rangox,np.amax(resultado[1])+rangox , np.amin(resultado[0])-rangoy,np.amax(resultado[0]) +rangoy])  
        plt.show()


class Red_mapeo_caracteristicas:
    def __init__(self, inputs, targets, fila_mapa, columna_mapa):
        self.cantidad_neuronas_entrada = np.shape(inputs)[1]
        self.cantidad_neuronas_salida = np.shape(targets)[1]
        self.cantidad_instancias_dataset = np.shape(inputs)[0]
        self.fila_mapa = fila_mapa
        self.columna_mapa = columna_mapa
        self.inputs_entrenamiento = inputs
        self.targets_entrenamiento = targets
        self.weights = np.random.normal(size=(self.cantidad_neuronas_entrada, self.fila_mapa * self.columna_mapa),scale= 1/np.sqrt(self.cantidad_neuronas_entrada))
        np.seterr(all='raise')

    def entrenar(self, learning_rate):
        for i in range(1000):
            for instancia in self.inputs_entrenamiento:
                y = activacion(instancia)
                correccion(x,y,i)


    #Falta debugear y ver que todas las cuentas esten bien
    def activacion(x):
        y_raya = np.linalg.norm(x - self.weights)
        y = (y_raya == np.amin(y_raya))*1.0
        return y

    def correccion(x,y,epoca):
        j_ast = np.nonzero(y)
        D = delta_func(j_ast,epoca)
        delta_weights = self.learning_rate * D *(x - self.weights)
        self.weights += delta_weights

    def delta_func(j_ast,epoca):
        rta = np.zeros(self.cantidad_neuronas_entrada)
        m = self.fila_mapa * self.columna_mapa
        for j in range(m)
            np.exp(-np.norm(P(j)-P(j_ast))**2/(2.0*variance(epoca)**2))

    def P(j):
        return [j/self.columna_mapa, np.mod(j,self.columna_mapa)

    def variance(epoca):
        return (self.columna_mapa/2.0)* epoca**(-1.0/3.0)
