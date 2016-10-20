import numpy as np
import matplotlib.pyplot as plt

class Red_hebbs:
    def __init__(self, inputs, targets, cantidad_componentes_principales):
        self.cantidad_neuronas_entrada = np.shape(inputs)[1]
        self.cantidad_neuronas_salida = cantidad_componentes_principales
        self.cantidad_instancias_dataset = np.shape(inputs)[0]

        self.inputs_entrenamiento = inputs
        self.targets_entrenamiento = targets

        self.weights = np.random.normal(size=(self.cantidad_neuronas_entrada, self.cantidad_neuronas_salida),scale= 1/np.sqrt(self.cantidad_neuronas_entrada))

        np.seterr(all='raise')

    def entrenar(self,learning_rate):
        for i in range(100):
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
        plt.scatter(resultado.T[0],resultado.T[1])
        #plt.axis([np.amin(resultado[1])-rangox,np.amax(resultado[1])+rangox , np.amin(resultado[0])-rangoy,np.amax(resultado[0]) +rangoy])  
        plt.show()


class Red_mapeo_caracteristicas:
    def __init__(self, inputs, targets, fila_mapa, columna_mapa):
        self.cantidad_neuronas_entrada = np.shape(inputs)[1]
        self.cantidad_instancias_dataset = np.shape(inputs)[0]
        self.fila_mapa = fila_mapa
        self.columna_mapa = columna_mapa
        self.inputs_entrenamiento = inputs
        self.targets_entrenamiento = targets
        self.weights = np.random.normal(size=(self.cantidad_neuronas_entrada, self.fila_mapa * self.columna_mapa),scale= 1/np.sqrt(self.cantidad_neuronas_entrada))
        np.seterr(all='raise')

    def entrenar(self, learning_rate):
        for i in range(100):
            for instancia in self.inputs_entrenamiento:
                y = self.activacion(instancia)
                self.correccion(instancia,y,i+1,learning_rate)
            print i


    #Falta debugear y ver que todas las cuentas esten bien
    def activacion(self, x):
        y_raya = np.linalg.norm(x - self.weights.T,axis=0)
        y = (y_raya == np.amin(y_raya))*1.0
        return y

    def correccion(self,x,y,epoca,learning_rate):
        #podria tener mas de un elemento en 1?
        j_ast = np.nonzero(y)[0][0]
        D = self.delta_func(j_ast,epoca)
        delta_weights = learning_rate * np.dot((x - self.weights.T),D )
        self.weights += delta_weights
        return

    def delta_func(self,j_ast,epoca):
        rta = np.zeros(self.cantidad_neuronas_entrada)
        m = self.fila_mapa * self.columna_mapa
        varianza = self.variance(epoca)
        P_j = self.P(j_ast)
        for j in range(m):
            np.exp(-np.linalg.norm(self.P(j)- P_j)**2/(2.0*varianza**2))
        return rta

    def P(self,j):
        return np.array([j / self.columna_mapa, np.mod(j,self.columna_mapa)])

    def variance(self,epoca):
        return (self.columna_mapa/2.0)* epoca**(-1.0/3.0)

    def testear(self,instancias):
        for instancia in instancias:
            index = np.argmax(np.dot(instancia,self.weights))
            print np.dot(instancia,self.weights)
            punto = [(index/self.fila_mapa) +1, (index%self.columna_mapa) +1]
            print punto
            plt.scatter(punto[0],punto[1])
        plt.axis([0,self.fila_mapa , 0,self.columna_mapa])
        plt.show()