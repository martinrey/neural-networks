import numpy as np


# Clase abstracta
class Funcion(object):
    def evaluar_en(self, punto):
        raise NotImplementedError

    def derivar_y_evaluar_en(self, punto):
        raise NotImplementedError


# Clase concreta
class SigmoideaLogistica(Funcion):
    def __init__(self, cte):
        self._cte = cte

    def evaluar_en(self, punto):
        try:
            resultado = 1.0 / (1.0 + np.exp(self._cte * punto))
        except FloatingPointError:
            resultado = 0
        return resultado

    def derivar_y_evaluar_en(self, punto):
        return self._cte * self.evaluar_en(punto) * (1.0 - self.evaluar_en(punto))


# Clase concreta
class Identidad(Funcion):
    def evaluar_en(self, punto):
        return punto

    def derivar_y_evaluar_en(self, punto):
        return punto
