import numpy as np


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
        nueva_capa = self.__class__(self._cantidad_neuronas, self._funcion_activacion)
        valores = np.zeros(self._cantidad_neuronas)
        for i in range(self.cantidad_neuronas()):
            valores[i] = self._funcion_activacion.derivar_y_evaluar_en(self._valores[i])
        nueva_capa.set_valores(valores)
        return nueva_capa

    def evaluar(self):
        nueva_capa = self.__class__(self._cantidad_neuronas, self._funcion_activacion)
        valores = np.zeros(self._cantidad_neuronas)
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
    def __init__(self, capas, algoritmo_de_aprendizaje):
        # por cuestiones de seguridad al ocurrir un overflow se lanza error
        np.seterr(over='raise')
        self._capas = capas
        self._matrices = []
        self._algoritmo_de_aprendizaje = algoritmo_de_aprendizaje

    def cantidad_de_capas(self):
        return len(self._capas)

    def capa_numero(self, numero_capa):
        return self._capas[numero_capa]

    def cantidad_de_matrices_de_pesos(self):
        return self.cantidad_de_capas() - 1

    def matriz_de_pesos_numero(self, numero_matriz):
        return self._matrices[numero_matriz]

    def set_matriz_de_peso(self, numero_matriz, matriz):
        self._matrices[numero_matriz] = matriz

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

    def entrenar(self, inputs, clasificaciones):
        for i in range(5):
            # TODO: ver si hacerlo como batch, mini-batch, etc
            for input, clasificacion in zip(inputs, clasificaciones):
                resultado_forwardeo = self._forward_propagation(input)
                self._algoritmo_de_aprendizaje.ejecutar(clasificacion, resultado_forwardeo, red_neuronal=self,
                                                        coeficiente_aprendizaje=1)
