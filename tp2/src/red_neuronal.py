import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

class Red_hebbs:
    #TODO: mode usa oja o sanjer
    def __init__(self, inputs, cantidad_componentes_principales, mode):
        self.cantidad_neuronas_entrada = np.shape(inputs)[1]
        self.cantidad_neuronas_salida = cantidad_componentes_principales
        self.cantidad_instancias_dataset = np.shape(inputs)[0]
        self.inputs_entrenamiento = inputs
        self.weights = np.random.normal(size=(self.cantidad_neuronas_entrada, self.cantidad_neuronas_salida),scale= 1/np.sqrt(self.cantidad_neuronas_entrada))
        np.seterr(over='raise')
        self.mode = mode

    def entrenar(self,learning_rate, iteraciones = 10000):
        U = np.triu(np.ones((self.cantidad_neuronas_salida,self.cantidad_neuronas_salida) ))
        for iteracion in range(iteraciones):
            for instancia in self.inputs_entrenamiento:
                if self.mode == 'oja':
                    y = np.dot(instancia, self.weights)
                    x_raya = np.dot(y,self.weights.T)
                    delta_weights = learning_rate * y * np.array([instancia - x_raya]).T
                elif self.mode == 'sanjer':
                    y = np.dot(instancia, self.weights)
                    x_raya = np.multiply( np.array([y]).T , U)
                    x_raya = np.dot(self.weights, x_raya)
                    delta_weights = learning_rate * (np.array([instancia]).T - x_raya) * y
                self.weights = self.weights + delta_weights
            if (iteracion * 100) % iteraciones == 0:
                print "Completo: ", (iteracion * 100)/iteraciones, "%"

    def testear(self,inputs,targets_entrenamiento):
        resultado = np.dot(inputs, self.weights)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(resultado.T[0],resultado.T[1],resultado.T[2],c=targets_entrenamiento)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


class Red_mapeo_caracteristicas:
    #Optimizacion Posible: Los imputs estan formados de vectores esparsos (muchos ceros), utilizar syphi.sparce para mejorar performance
    def __init__(self, inputs, fila_mapa, columna_mapa,):
        self.cantidad_neuronas_entrada = np.shape(inputs)[1]
        self.cantidad_instancias_dataset = np.shape(inputs)[0]
        self.fila_mapa = fila_mapa
        self.columna_mapa = columna_mapa
        self.inputs_entrenamiento = inputs
        self.weights = np.random.normal(size=(self.cantidad_neuronas_entrada, self.fila_mapa * self.columna_mapa),scale= 1/np.sqrt(self.cantidad_neuronas_entrada))
        np.seterr(over='raise')
        #para optimizar creo esto aca
        self.j = np.arange(self.cantidad_neuronas_entrada*1.0)

    def entrenar(self, learning_rate,iteraciones = 100):
        for i in range(iteraciones):
            for instancia in self.inputs_entrenamiento:
                y = self.activacion(instancia)
                self.correccion(instancia,y,i+1,learning_rate)
            if (i * 100) % iteraciones == 0:
                print "Completo: ", (i * 100)/iteraciones, "%" 


    #Falta debugear y ver que todas las cuentas esten bien
    def activacion(self, x):
        y_raya = np.linalg.norm(x - self.weights.T,axis=1)
        y = (y_raya == np.amin(y_raya))*1.0
        return y

    def correccion(self,x,y,epoca,learning_rate):
        #podria tener mas de un elemento en 1?
        j_ast = np.nonzero(y)
        j_ast = j_ast[0][0]
        D = self.delta_func(j_ast,epoca)
        #Cambiado el learning rate adaptativo se deberian obtener diferentes resultados
        delta_weights = learning_rate.calcular(epoca) * np.dot((x - self.weights.T),D )
        self.weights += delta_weights
        return

    def delta_func(self,j_ast,epoca):
        m = self.fila_mapa * self.columna_mapa
        varianza = self.variance(epoca)
        P_j = self.P(j_ast)
        return (np.exp(-np.linalg.norm(self.P(self.j).T - P_j,axis=1)**2)/(2.0*varianza**2))

    def P(self,j):
        return np.array([j / self.columna_mapa, np.mod(j,self.columna_mapa)])

    def variance(self,epoca):
        return (self.columna_mapa/2.0)* epoca**(-1.0/3.0)

    def testear(self,instancias, clasificaciones):
        colores = cm.rainbow(np.linspace(0, 1, 9))
        test_set = zip(instancias,clasificaciones)
        for (instancia,clasificacion) in test_set:
            index = np.argmax(np.dot(instancia,self.weights))
            punto = [(index/self.fila_mapa) +1, (index%self.columna_mapa) +1]
            print punto,clasificacion
            plt.scatter(punto[0],punto[1], color=colores[clasificacion])
        plt.axis([0,self.fila_mapa , 0,self.columna_mapa])
        plt.show()