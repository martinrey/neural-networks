from funcion import SigmoideaLogistica, Identidad, Tanh
from lector_de_instancias import LectorDeInstancias
from red_neuronal import PerceptronMulticapa, CapaInterna, CapaSalida
from adapters import InstanciaAPerceptronAdapter
import numpy as np


def normalizar(instancias):
    cantidad_de_instancias = len(instancias)

    medias = np.mean(instancias,axis=0)
    varianza_muestral = np.var(instancias,axis=0)
    cantidad_de_atributos = len(medias)

    for i in range(cantidad_de_instancias):
        for j in range(cantidad_de_atributos):
            instancias[i][j] = (instancias[i][j]-medias[j] +1)/varianza_muestral[j]
    return instancias

if __name__ == "__main__":
    tanh = Tanh()
    identidad = Identidad()
    sigmoidea = SigmoideaLogistica(cte=1)

    #WARNING: poner funcion de activiacion=identidad hace que diverja todo al chori
    capa_1 = CapaInterna(cantidad_neuronas=10, funcion_activacion=sigmoidea)
    capa_2 = CapaInterna(cantidad_neuronas=100, funcion_activacion=sigmoidea)
    capa_3 = CapaInterna(cantidad_neuronas=50, funcion_activacion=sigmoidea)
    capa_4 = CapaInterna(cantidad_neuronas=20, funcion_activacion=sigmoidea)
    capa_5 = CapaSalida(cantidad_neuronas=1, funcion_activacion=sigmoidea)

    capas = [capa_1, capa_2, capa_5]

    perceptron_multicapa = PerceptronMulticapa(capas)

    lector_de_instancias = LectorDeInstancias(archivo='tp1_ej1_training.csv')
    conjunto_de_instancias_de_entrenamiento = lector_de_instancias.leer()
    conjunto_de_instancias_vectorizadas = []
    clasificaciones = []
    for instancia in conjunto_de_instancias_de_entrenamiento:
        instancia_vectorizada, clasificacion = InstanciaAPerceptronAdapter().adaptar_esto(instancia)
        conjunto_de_instancias_vectorizadas.append(instancia_vectorizada)
        clasificaciones.append(clasificacion)
    conjunto_de_instancias_vectorizadas_normalizadas = normalizar(conjunto_de_instancias_vectorizadas)
    perceptron_multicapa.inicializar_pesos(len(conjunto_de_instancias_vectorizadas))
    perceptron_multicapa.entrenar(conjunto_de_instancias_vectorizadas_normalizadas, clasificaciones)


    
