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
            resultado = 1.0 / (1.0 + np.exp(-1.0 * self._cte * punto))
        except FloatingPointError:
            # print punto
            if (punto > 0.0):
                resultado = 1.0
            else:
                resultado = 0.0

        return resultado

    def derivar_y_evaluar_en(self, punto):
        return self._cte * self.evaluar_en(punto) * (1.0 - self.evaluar_en(punto))


# Clase concreta
class Identidad(Funcion):
    def evaluar_en(self, punto):
        return punto

    def derivar_y_evaluar_en(self, punto):
        return 1.0


# Clase concreta
class Tanh(Funcion):
    def evaluar_en(self, punto):
        return np.tanh(punto)

    def derivar_y_evaluar_en(self, punto):
        return 1.0 - (np.tanh(punto) ** 2)
