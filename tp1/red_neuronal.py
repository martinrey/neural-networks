import numpy as np


# Clase abstracta
class Capa(object):
    def __init__(self, cantidad_neuronas, funcion_activacion):
        self._cantidad_neuronas = cantidad_neuronas
        self._funcion_activacion = funcion_activacion
        self._valores = np.zeros(cantidad_neuronas)

    def cantidad_neuronas(self):
        return self._cantidad_neuronas

    def evaluar_en_derivada(self):
        nueva_capa = self.__class__(self._cantidad_neuronas ,self._funcion_activacion )
        valores = np.zeros(self._cantidad_neuronas)
        for i in range(self.cantidad_neuronas()):
            valores[i] = self._funcion_activacion.derivar_y_evaluar_en(self._valores[i])
        nueva_capa.set_valores(valores)
        return nueva_capa

    def evaluar(self):
        nueva_capa = self.__class__(self._cantidad_neuronas ,self._funcion_activacion )
        valores = np.zeros(self._cantidad_neuronas)
        for i in range(self.cantidad_neuronas()):
            valores[i] = self._funcion_activacion.evaluar_en(self._valores[i])
        nueva_capa.set_valores(valores)
        return nueva_capa

    def valores(self):
        return self._valores


# Clase concreta
class CapaInterna(Capa):
    def __init__(self, cantidad_neuronas, funcion_activacion):
        super(CapaInterna, self).__init__(cantidad_neuronas, funcion_activacion)
        self._valores = np.hstack((self._valores, [-1]))

    def set_valores(self, valores):
        self._valores[:-1] = valores
        return self


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
        self._capas = capas
        self._matrices = []

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
            self._matrices.append(
                np.random.rand(self.capa_numero(indice_capa).cantidad_neuronas() + 1,
                               self.capa_numero(indice_capa + 1).cantidad_neuronas())
            )

    def _forward_propagation(self, input):
        self._capas[0] = self.capa_numero(0).set_valores(input).evaluar()
        for indice_capa in range(self.cantidad_de_capas() - 1):
            np_dot = np.dot(self.capa_numero(indice_capa).valores(), self.matriz_de_pesos_numero(indice_capa))
            self._capas[indice_capa + 1] = self.capa_numero(indice_capa + 1).set_valores(np_dot).evaluar()
        return self.capa_numero(self.cantidad_de_capas() - 1).valores()

    #usando el algoritmo pag. 120 del hertz
    def _back_propagation(self, clasificacion, resultado_forwardeo):
        coeficiente_aprendisaje = 1
        #Paso 4 (1-3 son forward)
        derivada_ultima_capa = self.capa_numero(self.cantidad_de_capas() - 1).evaluar_en_derivada().valores()
        diferencia_respuestas_esperada_obtenida = np.subtract(clasificacion, resultado_forwardeo)
        delta_ultima_capa = np.multiply(derivada_ultima_capa,diferencia_respuestas_esperada_obtenida)
        deltas = []
        deltas.append(delta_ultima_capa)
        #paso 5
        for i in range( self.cantidad_de_capas() - 2 ,-1 ,-1):
            derivada_capa_i = self.capa_numero(i).evaluar_en_derivada().valores()
            derivada_capa_i_mas_uno = deltas[-1]
            #multiplico fila a fila, pero hay problemas con las dimenciones otra vez
            #mismo resultado que con:
            #producto_matriz_y_vector_delta = np.dot( derivada_capa_i_mas_uno, np.transpose(self.matriz_de_pesos_numero(i)))
            #estan mal las dimensiones?
            producto_matriz_y_vector_delta = []
            # print "size vec"
            # print derivada_capa_i_mas_uno.size
            # print "---Cols:---"
            cantidad_de_columnas = self.matriz_de_pesos_numero(i).size/self.matriz_de_pesos_numero(i)[0].size
            # print cantidad_de_columnas
            # print "---Fils:---"
            # print self.matriz_de_pesos_numero(i)[0].size
            
            for cols in range(cantidad_de_columnas ):
                sumatoria_col_j = 0
                cantidad_de_filas = self.matriz_de_pesos_numero(i)[cols].size
                for fils in range(cantidad_de_filas):
                    sumatoria_col_j += self.matriz_de_pesos_numero(i)[cols][fils] * derivada_capa_i_mas_uno[fils]
                producto_matriz_y_vector_delta.append(sumatoria_col_j)
            delta_capa_i = np.multiply(derivada_capa_i, producto_matriz_y_vector_delta)
            deltas.append(delta_capa_i)
        #paso 6
        for m in range(self.cantidad_de_capas() - 1):
            filas = deltas[self.cantidad_de_capas() - 1 -m].size
            columnas = self.capa_numero(m+1).valores().size
            delta_matriz = np.zeros(( filas, columnas))
            for i in range(filas):
                for k in range(columnas):
                    delta_matriz[i][k] = coeficiente_aprendisaje * deltas[self.cantidad_de_capas() - 1 -m][i]*self.capa_numero(m+1).valores()[k]
            #Problema, no dan las dimenciones
            print "Cantidad De filas en la matriz delta:"
            print delta_matriz[0].size
            print "Cantidad De filas en la matriz de pesos:"
            print self.matriz_de_pesos_numero(m)[0].size
            print "Wtf"
        pass

    def entrenar(self, inputs, clasificaciones):
        # TODO: ver si hacerlo como batch, mini-batch, etc
        for input, clasificacion in zip(inputs, clasificaciones):
            resultado_forwardeo = self._forward_propagation(input)
            #print resultado_forwardeo
            self._back_propagation(clasificacion, resultado_forwardeo)


