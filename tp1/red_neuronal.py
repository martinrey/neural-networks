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
        return self._funcion_activacion.derivar_y_evaluar_en(self._valores)

    def evaluar(self):
        return self._funcion_activacion.evaluar_en(self._valores)


# Clase concreta
class CapaInterna(Capa):
    def __init__(self, cantidad_neuronas, funcion_activacion):
        super(CapaInterna, self).__init__(cantidad_neuronas, funcion_activacion)
        self._valores = np.hstack((self._valores, [-1]))

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
        self._delta_matrices = []
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
        for indice_capa in range(self.cantidad_de_capas() - 1):
            self._delta_matrices_old.append(np.zeros((self.capa_numero(indice_capa).cantidad_neuronas(),
                                                  self.capa_numero(indice_capa + 1).cantidad_neuronas())))
            self._matrices.append(
                np.random.normal(size=(self.capa_numero(indice_capa).cantidad_neuronas(),
                               self.capa_numero(indice_capa + 1).cantidad_neuronas()),scale=1.0/np.sqrt(cantidad_de_instancias))
            )


    def inicializar_pesos_mat_delta(self, cantidad_de_instancias):
        np.random.seed(10)
        for indice_capa in range(self.cantidad_de_capas() - 1):
            self._delta_matrices.append(np.zeros((self.capa_numero(indice_capa).cantidad_neuronas(),
                                                  self.capa_numero(indice_capa + 1).cantidad_neuronas())))


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
        delta_ultima_capa = np.multiply(derivada_ultima_capa, diferencia_respuestas_esperada_obtenida)
        deltas = []
        deltas.append(delta_ultima_capa)
        # paso 5
        for i in range(self.cantidad_de_capas() - 2, -1, -1):
            derivada_capa_i = self.capa_numero(i).evaluar_en_derivada()
            derivada_capa_i_mas_uno = deltas[-1]
            producto_matriz_y_vector_delta = np.dot(derivada_capa_i_mas_uno,
                                                    np.transpose(self.matriz_de_pesos_numero(i)))
            delta_capa_i = np.multiply(derivada_capa_i, producto_matriz_y_vector_delta)
            deltas.append(delta_capa_i)
        # paso 6
        for m in range(self.cantidad_de_capas() - 1):
            self._delta_matrices[m] = self._delta_matrices[m] + (coeficiente_aprendisaje * np.outer(deltas[self.cantidad_de_capas() - 1 - m], self.capa_numero(m + 1).evaluar()))
        return

    def entrenar(self, inputs, clasificaciones):
        instancias = zip(inputs, clasificaciones)
        test, entrenamiento = self.split(instancias, 1.0 / 5)
        self.inicializar_pesos(len(entrenamiento))
        print "Iniciando Aprendisaje"
        norma_del_error = 1000
        b = 0.9
        a = 0.00001
        coeficiente_aprendisaje = 0.001
        momentum = 0.1
        contador_de_errores = 0
        # Minibatch con instancias randomizadas:
        for i in range(500):
            error = []
            shuffle(entrenamiento)
            self.inicializar_pesos_mat_delta(len(entrenamiento))
            for input, clasificacion in entrenamiento:
                resultado_forwardeo = self._forward_propagation(input)
                self._back_propagation(clasificacion, resultado_forwardeo, error, coeficiente_aprendisaje, momentum)
                #print "%f %f" % (resultado_forwardeo, clasificacion)
            for m in range(self.cantidad_de_capas() - 1):
            	self._matrices[m] = self._matrices[m] + self._delta_matrices[m] + (momentum * self._delta_matrices_old[m])
                self._delta_matrices_old[m] = self._delta_matrices[m]
            #print "Iteracion: %d, Norma del errpr: %f" % (i, np.linalg.norm(error))
            if (norma_del_error - np.linalg.norm(error) < 0):
                #print "Mas error, aflojo"
                coeficiente_aprendisaje -= b * coeficiente_aprendisaje
                if(contador_de_errores > 3):
                    #print "Cantidad de errores alcanzado, Fin de entrenamiento!"
                    break
                else:
                    contador_de_errores = contador_de_errores + 1
            else:
                coeficiente_aprendisaje += a
            norma_del_error = np.linalg.norm(error)
        # testeo que tan buenos resultados obtengo:
        #print "Inicio Testeo de resutados:"
        asiertos  = 0
        fallos = 0
        for input, clasificacion in test:
            resultado_forwardeo = self._forward_propagation(input)
            #print "Prediccion: %f Verdad: %f" % (resultado_forwardeo, clasificacion)
            if(resultado_forwardeo > 0.0):
                resultado_forwardeo = 1.0
            if(resultado_forwardeo < 0.0):
                resultado_forwardeo = -1.0
            if(clasificacion == resultado_forwardeo):
                asiertos = asiertos + 1
            else:
                fallos = fallos + 1

        #print "Asiertos: %d Fallos: %d" % (asiertos, fallos)
        porcentaje_aciertos = ((1.0 * asiertos/ (fallos+asiertos)) * 100.0)
        print "Porcentaje De Aciertos: %f" % porcentaje_aciertos
        return porcentaje_aciertos

            # valor numero entre 0 y 1

    def split(self, inputs, valor):
        proporcion = len(inputs) * valor
        return inputs[:int(proporcion)], inputs[int(proporcion):]
