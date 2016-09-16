import numpy as np
from random import shuffle

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
        nueva_capa = self.__class__(self._cantidad_neuronas ,self._funcion_activacion )
        valores = np.zeros(self.cantidad_neuronas())
        for i in range(self.cantidad_neuronas()):
            valores[i] = self._funcion_activacion.derivar_y_evaluar_en(self._valores[i])
        nueva_capa.set_valores(valores)
        return nueva_capa

    def evaluar(self):
        nueva_capa = self.__class__(self._cantidad_neuronas ,self._funcion_activacion )
        valores = np.zeros(self.cantidad_neuronas())
        for i in range(self.cantidad_neuronas()):
            valores[i] = self._funcion_activacion.evaluar_en(self._valores[i])
        nueva_capa.set_valores(valores)
        return nueva_capa



# Clase concreta
class CapaInterna(Capa):
    def __init__(self, cantidad_neuronas, funcion_activacion):
        super(CapaInterna, self).__init__(cantidad_neuronas, funcion_activacion)
        self._valores = np.hstack((self._valores, [-1]))

    def set_valores(self, valores):
    	if(len(valores) == self._cantidad_neuronas):
        	self._valores[:-1] = valores
        else:
        	if(len(valores) == self._cantidad_neuronas + 1):
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
        #por cuestiones de seguridad al ocurrir un overflow se lanza error
        np.seterr(over='raise')
        self._capas = capas
        self._matrices = []
        self._delta_matrices = []

    def cantidad_de_capas(self):
        return len(self._capas)

    def capa_numero(self, numero_capa):
        return self._capas[numero_capa]

    def cantidad_de_matrices_de_pesos(self):
        return self.cantidad_de_capas() - 1

    def matriz_de_pesos_numero(self, numero_matriz):
        return self._matrices[numero_matriz]

    def inicializar_pesos(self):
        np.random.seed(1)
        for indice_capa in range(self.cantidad_de_capas() - 1):
            self._delta_matrices.append( np.zeros((self.capa_numero(indice_capa).cantidad_neuronas(), self.capa_numero(indice_capa + 1).cantidad_neuronas())))
            self._matrices.append(
                np.random.rand(self.capa_numero(indice_capa).cantidad_neuronas(),
                               self.capa_numero(indice_capa + 1).cantidad_neuronas())*0.0000001
            )

    def _forward_propagation(self, input):
        self._capas[0] = self.capa_numero(0).set_valores(input).evaluar()
        for indice_capa in range(self.cantidad_de_capas() - 1):
            np_dot = np.dot(self.capa_numero(indice_capa).valores(), self.matriz_de_pesos_numero(indice_capa))
            self._capas[indice_capa + 1] = self.capa_numero(indice_capa + 1).set_valores(np_dot).evaluar()
        #print self.capa_numero(self.cantidad_de_capas() - 1).valores()
        return self.capa_numero(self.cantidad_de_capas() - 1).valores()

    #usando el algoritmo pag. 120 del hertz
    def _back_propagation(self, clasificacion, resultado_forwardeo, error, coeficiente_aprendisaje, momentum=0.9):
        #Paso 4 (1-3 son forward)
        derivada_ultima_capa = self.capa_numero(self.cantidad_de_capas() - 1).evaluar_en_derivada().valores()
        #print clasificacion
        diferencia_respuestas_esperada_obtenida = np.subtract(clasificacion, resultado_forwardeo)
       	#print diferencia_respuestas_esperada_obtenida
        error.append(diferencia_respuestas_esperada_obtenida)
        delta_ultima_capa = np.multiply(derivada_ultima_capa + 0.1	,diferencia_respuestas_esperada_obtenida)
        deltas = []
        deltas.append(delta_ultima_capa)
        #paso 5
        for i in range( self.cantidad_de_capas() - 2 ,-1 ,-1):
            derivada_capa_i = self.capa_numero(i).evaluar_en_derivada().valores()
            derivada_capa_i_mas_uno = deltas[-1]
            #multiplico fila a fila, pero hay problemas con las dimenciones otra vez
            producto_matriz_y_vector_delta = np.dot( derivada_capa_i_mas_uno, np.transpose(self.matriz_de_pesos_numero(i)))
            #mismo resultado que con:
            # producto_matriz_y_vector_delta = []
            # cantidad_de_columnas = self.matriz_de_pesos_numero(i).size/self.matriz_de_pesos_numero(i)[0].size
            # for cols in range(cantidad_de_columnas ):
            #     sumatoria_col_j = 0
            #     cantidad_de_filas = self.matriz_de_pesos_numero(i)[cols].size
            #     for fils in range(cantidad_de_filas):
            #         sumatoria_col_j += self.matriz_de_pesos_numero(i)[cols][fils] * derivada_capa_i_mas_uno[fils]
            #     producto_matriz_y_vector_delta.append(sumatoria_col_j)
            delta_capa_i = np.multiply(derivada_capa_i, producto_matriz_y_vector_delta)
            deltas.append(delta_capa_i)
        #paso 6
        for m in range(self.cantidad_de_capas() - 1):
            filas = deltas[self.cantidad_de_capas() - 1 -m].size
            #en realidad existe una columna mas que no se usa, sabe dios para que es
            columnas = self.capa_numero(m+1).cantidad_neuronas()
            for i in range(filas):
                for k in range(columnas):
                    self._delta_matrices[m][i][k] = coeficiente_aprendisaje * deltas[self.cantidad_de_capas() - 1 -m][i]*self.capa_numero(m+1).valores()[k] + momentum * self._delta_matrices[m][i][k] 
            #print self._delta_matrices
            self._matrices[m] = np.add(self.matriz_de_pesos_numero(m),self._delta_matrices[m])
        return

    def entrenar(self, inputs, clasificaciones):
        instancias = zip(inputs, clasificaciones)
        test, entrenamiento = self.split(instancias, 1.0/3)
        print "Iniciando Aprendisaje"
        #Minibatch con instancias randomizadas:
        norma_del_error = 1000
        b = 0.5
        a = 0.0005
        coeficiente_aprendisaje = 0.05
        momentum = 0.9
        for i in range(50):
            error = []
            shuffle(entrenamiento)
            for input, clasificacion in entrenamiento:
                resultado_forwardeo = self._forward_propagation(input)
                self._back_propagation(clasificacion, resultado_forwardeo, error, coeficiente_aprendisaje,momentum)
            print "Iteracion: %d, Norma del errpr: %f" % (i , np.linalg.norm(error))
            if(norma_del_error - np.linalg.norm(error) < 0):	
            	print "Mas error, aflojo"
            	momentum = 0
            	coeficiente_aprendisaje -= b*coeficiente_aprendisaje
            else:
            	momentum = 0.9
            	coeficiente_aprendisaje += a
            norma_del_error = np.linalg.norm(error)

        #testeo que tan buenos resultados obtengo:
        print "Inicio Testeo de resutados:"
        for input, clasificacion in test:
            resultado_forwardeo = self._forward_propagation(input)
            print "Prediccion: %f Verdad: %f" % (resultado_forwardeo, clasificacion) 



    #valor numero entre 0 y 1
    def split(self, inputs, valor):
	    proporcion = len(inputs)*valor
	    return inputs[:int(proporcion)], inputs[int(proporcion):]
