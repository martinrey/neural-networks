import numpy as np


class AlgoritmoDeAprendizaje(object):
    def ejecutar(self):
        raise NotImplementedError


class BackPropagation(AlgoritmoDeAprendizaje):
    def ejecutar(self, clasificacion, resultado_forwardeo, red_neuronal, coeficiente_aprendizaje):
        # Paso 4 (1-3 son forward)
        derivada_ultima_capa = red_neuronal.capa_numero(
            red_neuronal.cantidad_de_capas() - 1).evaluar_en_derivada().valores()
        diferencia_respuestas_esperada_obtenida = np.subtract(clasificacion, resultado_forwardeo)
        delta_ultima_capa = np.multiply(derivada_ultima_capa, diferencia_respuestas_esperada_obtenida)
        deltas = []
        deltas.append(delta_ultima_capa)
        # paso 5
        for i in range(red_neuronal.cantidad_de_capas() - 2, -1, -1):
            derivada_capa_i = red_neuronal.capa_numero(i).evaluar_en_derivada().valores()
            derivada_capa_i_mas_uno = deltas[-1]
            # multiplico fila a fila, pero hay problemas con las dimenciones otra vez
            # mismo resultado que con:
            # producto_matriz_y_vector_delta = np.dot( derivada_capa_i_mas_uno, np.transpose(self.matriz_de_pesos_numero(i)))
            producto_matriz_y_vector_delta = []
            cantidad_de_columnas = red_neuronal.matriz_de_pesos_numero(i).size / \
                                   red_neuronal.matriz_de_pesos_numero(i)[0].size
            for cols in range(cantidad_de_columnas):
                sumatoria_col_j = 0
                cantidad_de_filas = red_neuronal.matriz_de_pesos_numero(i)[cols].size
                for fils in range(cantidad_de_filas):
                    sumatoria_col_j += red_neuronal.matriz_de_pesos_numero(i)[cols][fils] * \
                                       derivada_capa_i_mas_uno[fils]
                producto_matriz_y_vector_delta.append(sumatoria_col_j)
            delta_capa_i = np.multiply(derivada_capa_i, producto_matriz_y_vector_delta)
            deltas.append(delta_capa_i)
        # paso 6
        for m in range(red_neuronal.cantidad_de_capas() - 1):
            filas = deltas[red_neuronal.cantidad_de_capas() - 1 - m].size
            # en realidad existe una columna mas que no se usa, sabe dios para que es
            columnas = red_neuronal.capa_numero(m + 1).cantidad_neuronas()
            delta_matriz = np.zeros((filas, columnas))
            for i in range(filas):
                for k in range(columnas):
                    delta_matriz[i][k] = coeficiente_aprendizaje * \
                                         deltas[red_neuronal.cantidad_de_capas() - 1 - m][i] * \
                                         red_neuronal.capa_numero(m + 1).valores()[k]
            red_neuronal.set_matriz_de_peso(m, np.add(red_neuronal.matriz_de_pesos_numero(m), delta_matriz))
