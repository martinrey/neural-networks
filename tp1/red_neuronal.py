import numpy as np


# Clase concreta
class Capa(object):
    def __init__(self, cantidad_neuronas, funcion_activacion):
        self._cantidad_neuronas = cantidad_neuronas
        self._funcion_activacion = funcion_activacion
        self._valores = np.zeros(cantidad_neuronas)

    def cantidad_neuronas(self):
        return self._cantidad_neuronas

    def set_valores(self, valores):
        self._valores = valores

    def valores(self):
        return self._valores

    def evaluar(self):
        # TODO: ver si se puede hacer con np.vectorize
        vector_evaluado = []
        for i in range(self.cantidad_neuronas()):
            vector_evaluado.append(self._funcion_activacion.evaluar_en(self._valores[i]))
        return vector_evaluado


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
        for indice_capa in range(self.cantidad_de_capas() - 1):
            self._matrices.append(
                np.random.rand(self.capa_numero(indice_capa).cantidad_neuronas(),
                               self.capa_numero(indice_capa + 1).cantidad_neuronas())
            )

    def _forward_propagation(self):
        for indice_capa in range(self.cantidad_de_capas() - 1):
            self.capa_numero(indice_capa + 1).set_valores(np.dot(self.capa_numero(indice_capa).evaluar(), self.matriz_de_pesos_numero(indice_capa)))

    def _back_propagation(self):
        pass

    def entrenar(self):
        self._forward_propagation()
        self._back_propagation()
