import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from collections import Counter
import matplotlib

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

    def entrenar(self,learning_rate, iteraciones = 100):
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

    def testear(self,inputs,targets_entrenamiento, title, show_graphic=1):
        resultado = np.dot(inputs, self.weights)
        print resultado
        if show_graphic:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(resultado.T[0],resultado.T[1],resultado.T[2],c=targets_entrenamiento)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title(title)
            plt.show()

    def save_net(self,string='red_mapeo_caracteristicas'):
        np.save(file=(string+".npy"), arr=self.weights)

    def load_net(self,string='red_mapeo_caracteristicas'):
        self.weights = np.load(file= string+".npy")


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
        self.array_P = np.array([[j / self.columna_mapa, np.mod(j,self.columna_mapa)] for j in range(self.fila_mapa * self.columna_mapa)])

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
        delta_weights = learning_rate.calcular(epoca) * D * (x - self.weights.T).T
        self.weights += delta_weights
        return

    def delta_func(self,j_ast,epoca):
        m = self.fila_mapa * self.columna_mapa
        varianza = self.variance(epoca)
        P_j_ast = self.P(j_ast)
        return (np.exp(-np.linalg.norm(self.array_P - P_j_ast,axis=1)**2)/(2.0*varianza**2))

    def P(self,j):
        return self.array_P[j]

    def variance(self,epoca):
        return (self.columna_mapa/2.0)* epoca**(-1.0/3.0)

    def testear(self,instancias, clasificaciones, title, show_graphic=1):
        colores = cm.rainbow(np.linspace(0, 1, 10))
        test_set = zip(instancias,clasificaciones)
        resultados = [[] for i in range(self.fila_mapa*self.columna_mapa)]
        grafico = np.zeros((self.fila_mapa,self.columna_mapa))
        for (instancia,clasificacion) in test_set:
            y = self.activacion(instancia)
            j_ast = np.nonzero(y)
            index = j_ast[0][0]
            resultados[index].append(clasificacion[0])
            punto = [(index/self.fila_mapa) +1, (index%self.columna_mapa) +1]
            print punto,clasificacion
        for index in range(len(resultados)):
            if resultados[index]:
                data = Counter(resultados[index])
                ganador = data.most_common(1)
                grafico[(index/self.fila_mapa), (index%self.columna_mapa)] = ganador[0][0]
            else:
                grafico[(index/self.fila_mapa), (index%self.columna_mapa)] = np.nan
        if show_graphic:
            #plt.axis([0,self.fila_mapa , 0,self.columna_mapa])
            #plt.title(title)
            cmap = matplotlib.cm.jet
            cmap.set_bad('black',1.)
            plt.matshow(grafico,cmap=cmap)
            plt.show()

    def save_net(self,string='red_mapeo_caracteristicas'):
        np.save(file=(string+".npy"), arr=self.weights)

    def load_net(self,string='red_mapeo_caracteristicas'):
        self.weights = np.load(file= string+".npy")