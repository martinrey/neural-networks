import numpy as np
from random import shuffle
import os


# Clase abstracta
class Capa(object):
    def __init__(self, cantidad_neuronas, funcion_activacion):
        self._cantidad_neuronas = cantidad_neuronas
        self._funcion_activacion = funcion_activacion
        self._valores = np.zeros(cantidad_neuronas)

    def cantidad_neuronas(self):
        return self._cantidad_neuronas

    def valores(self):
        return self._valores

    def evaluar_en_derivada(self):
        return self._funcion_activacion.derivar_y_evaluar_en(self._valores)

    def evaluar(self):
        return self._funcion_activacion.evaluar_en(self._valores)


# Clase concreta
class CapaInterna(Capa):
    def __init__(self, cantidad_neuronas, funcion_activacion):
        super(CapaInterna, self).__init__(cantidad_neuronas, funcion_activacion)
        self._valores = np.hstack((self._valores, [-1.0]))

    def set_valores(self, valores):
        if (len(valores) == self._cantidad_neuronas):
            self._valores[:-1] = valores
        else:
            if (len(valores) == self._cantidad_neuronas + 1):
                self._valores[:-1] = valores[:-1]
            else:
                raise error
        return self

    def cantidad_neuronas(self):
        return self._cantidad_neuronas + 1


# Clase concreta
class CapaSalida(Capa):
    def __init__(self, cantidad_neuronas, funcion_activacion):
        super(CapaSalida, self).__init__(cantidad_neuronas, funcion_activacion)

    def set_valores(self, valores):
        self._valores = valores
        return self


# Clase concreta
class PerceptronMulticapa(object):
    def __init__(self, capas):
        # por cuestiones de seguridad al ocurrir un overflow se lanza error
        np.seterr(over='raise')
        self._capas = capas
        self._matrices = []
        self._delta_matrices_old = []

    def cantidad_de_capas(self):
        return len(self._capas)

    def capa_numero(self, numero_capa):
        return self._capas[numero_capa]

    def cantidad_de_matrices_de_pesos(self):
        return self.cantidad_de_capas() - 1

    def matriz_de_pesos_numero(self, numero_matriz):
        return self._matrices[numero_matriz]

    def inicializar_pesos(self, cantidad_de_instancias):
        np.random.seed(10)
        self._matrices = []
        self._delta_matrices_old = []
        self._delta_matrices = []
        for indice_capa in range(self.cantidad_de_capas() - 1):
            self._delta_matrices_old.append(np.zeros((self.capa_numero(indice_capa).cantidad_neuronas(),
                                                  self.capa_numero(indice_capa + 1).cantidad_neuronas())))
            self._matrices.append(
                np.random.normal(size=(self.capa_numero(indice_capa).cantidad_neuronas(),
                               self.capa_numero(indice_capa + 1).cantidad_neuronas()),scale=1.0/np.sqrt(cantidad_de_instancias)))
            self._delta_matrices.append(np.zeros((self.capa_numero(indice_capa).cantidad_neuronas(),
                                                  self.capa_numero(indice_capa + 1).cantidad_neuronas())))

    def inicializar_matrices_delta(self,cantidad_de_instancias):
        self._delta_matrices = []
        for indice_capa in range(self.cantidad_de_capas() - 1):
            self._delta_matrices.append(0.1*
                np.random.normal(size=(self.capa_numero(indice_capa).cantidad_neuronas(),
                               self.capa_numero(indice_capa + 1).cantidad_neuronas()),scale=1.0/np.sqrt(cantidad_de_instancias)))


    def _forward_propagation(self, input):
        self._capas[0] = self.capa_numero(0).set_valores(input)
        for indice_capa in range(self.cantidad_de_capas() - 1):
            np_dot = np.dot(self.capa_numero(indice_capa).evaluar(), self.matriz_de_pesos_numero(indice_capa))
            self._capas[indice_capa + 1].set_valores(np_dot)
        return self.capa_numero(self.cantidad_de_capas()-1).evaluar()

    # usando el algoritmo pag. 120 del hertz
    def _back_propagation(self, clasificacion, resultado_forwardeo, error, coeficiente_aprendisaje, momentum=0.9):
        # Paso 4 (1-3 son forward)
        derivada_ultima_capa = self.capa_numero(self.cantidad_de_capas() - 1).evaluar_en_derivada()
        diferencia_respuestas_esperada_obtenida = np.subtract(clasificacion, resultado_forwardeo)
        error.append(diferencia_respuestas_esperada_obtenida)
        delta_ultima_capa = derivada_ultima_capa * diferencia_respuestas_esperada_obtenida
        #print derivada_ultima_capa
        deltas = []
        deltas.append(delta_ultima_capa)
        # paso 5
        for i in range(self.cantidad_de_capas() - 2, -1, -1):
            #i empieza en 2 y vaja a 1..0
            derivada_capa_i = self.capa_numero(i).evaluar_en_derivada()
            delta_capa_i_mas_uno = deltas[-1]
            deltas.append(delta_capa_i_mas_uno.dot(self.matriz_de_pesos_numero(i).T) * derivada_capa_i)
        deltas = list(reversed(deltas))
        # paso 6
        for m in range(self.cantidad_de_capas() -1):
            layer = np.atleast_2d(self.capa_numero(m).evaluar())
            delta = np.atleast_2d(deltas[m+1])
            self._delta_matrices[m] = (coeficiente_aprendisaje * layer.T.dot(delta))
        return

    def entrenar(self, inputs, clasificaciones,verbose=0, coeficiente_aprendisaje = 0.1):
        instancias = zip(inputs, clasificaciones)
        test, entrenamiento = self.split(instancias, 1.0/3 )
        self.inicializar_pesos(len(entrenamiento))
        if(verbose):
            print "Iniciando Aprendisaje"
        self.norma_del_error = 1000
        b = 0.5
        a = 0.001
        
        momentum = 0.9
        # Minibatch con instancias randomizadas:
        for i in range(700):
            error = []
            shuffle(entrenamiento)
            #if(i < 500):
                #self.inicializar_matrices_delta(len(entrenamiento))
            for input, clasificacion in entrenamiento:
                resultado_forwardeo = self._forward_propagation(input)
                self._back_propagation(clasificacion, resultado_forwardeo, error, coeficiente_aprendisaje)
                self.actualizar_matrices(momentum)
            coeficiente_aprendisaje = self.actualizar_learning_rate(coeficiente_aprendisaje,error,verbose=verbose)
            self.norma_del_error = np.linalg.norm(error)
            if(verbose):
                print "Iteracion: %d, Norma del errpr: %f" % (i, self.norma_del_error)
        return self.testear_resultados(test,verbose)
        

            # valor numero entre 0 y 1

    def split(self, inputs, valor):
        proporcion = len(inputs) * valor
        return inputs[:int(proporcion)], inputs[int(proporcion):]

    def actualizar_matrices(self, momentum):
        for m in range(self.cantidad_de_capas() - 1):
                self._matrices[m] = self._matrices[m] + self._delta_matrices[m] + (momentum * self._delta_matrices_old[m])
                self._delta_matrices_old[m] = self._delta_matrices[m]
    
    def actualizar_learning_rate(self, coeficiente_aprendisaje, error, a=0.01, b=0.9, verbose=0):
        if (self.norma_del_error - np.linalg.norm(error) < 0):
            if(verbose):
                print "Mas error, aflojo"
            if(coeficiente_aprendisaje > 0.000001):
                coeficiente_aprendisaje -= b * coeficiente_aprendisaje
        else:
            coeficiente_aprendisaje += a
        return coeficiente_aprendisaje

    def testear_resultados(self,test,verbose=0):
        if(verbose):
            print "Inicio Testeo de resutados:"
        asiertos  = 0
        fallos = 0
        for input, clasificacion in test:
            resultado_forwardeo = self._forward_propagation(input)
            #print "Prediccion: Verdad: "  
            #print resultado_forwardeo 
            #print clasificacion
            if(resultado_forwardeo > 0.0):
                resultado_forwardeo = 1.0
            if(resultado_forwardeo < 0.0):
                resultado_forwardeo = -1.0
            if(clasificacion == resultado_forwardeo):
                asiertos = asiertos + 1
            else:
                fallos = fallos + 1
        if(verbose):
            print "Asiertos: %d Fallos: %d" % (asiertos, fallos)
        porcentaje_aciertos = ((1.0 * asiertos/ (fallos+asiertos)) * 100.0)
        if(verbose):
            print "Porcentaje De Aciertos: %f" % porcentaje_aciertos
        return porcentaje_aciertos

    def save_net(self,string):
        if not os.path.exists(string):
            os.makedirs(string)
        for indice_capa in range(self.cantidad_de_capas() - 1):
            np.save(file=( string + "/" + string+str(indice_capa)+".npy"),arr = self.capa_numero(indice_capa))

    def load_net(self, string):
        path, dirs, files = os.walk(string + "/").next()
        file_count = len(files)
        self._matrices = []
        for indice_capa in range(file_count):
            self._matrices.append(np.load(file=( string + "/" + string + str(indice_capa)+".npy")))