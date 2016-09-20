from funcion import SigmoideaLogistica, Identidad, Tanh
from lector_de_instancias import LectorDeInstancias
from red_neuronal import PerceptronMulticapa, CapaInterna, CapaSalida
from adapters import InstanciaAPerceptronAdapter
import numpy as np


def normalizar(instancias):
    resutaldo = []

    primer_cuartil = np.percentile(instancias, 25,axis=0)
    tercer_cuartil = np.percentile(instancias, 75,axis=0)
    IRQ = tercer_cuartil - primer_cuartil
    outlier = IRQ * 1.5
    cota_inferior = primer_cuartil - outlier
    cota_superior = tercer_cuartil + outlier
    for inst in instancias:
        instancia_tiene_outliers = np.any(np.logical_or(np.less(inst, cota_inferior), np.greater(inst, cota_superior)))
        if(instancia_tiene_outliers == False):
            resutaldo.append(inst)
        else:
            print "outlier encontrado!"
            print np.logical_or(np.less(inst, cota_inferior), np.greater(inst, cota_superior))
    cantidad_de_instancias = len(resutaldo)
    medias = np.mean(resutaldo,axis=0)
    varianza_muestral = np.var(resutaldo,axis=0)
    cantidad_de_atributos = len(medias)
    for i in range(cantidad_de_instancias):
        resutaldo[i] = (resutaldo[i]-medias)/np.sqrt(varianza_muestral)
    return resutaldo

if __name__ == "__main__":
    tanh = Tanh()
    identidad = Identidad()
    sigmoidea = SigmoideaLogistica(cte=1)

    lector_de_instancias = LectorDeInstancias(archivo='tp1_ej1_training.csv')
    conjunto_de_instancias_de_entrenamiento = lector_de_instancias.leer()
    conjunto_de_instancias_vectorizadas = []
    clasificaciones = []
    for instancia in conjunto_de_instancias_de_entrenamiento:
        instancia_vectorizada, clasificacion = InstanciaAPerceptronAdapter().adaptar_esto(instancia)
        conjunto_de_instancias_vectorizadas.append(instancia_vectorizada)
        clasificaciones.append(clasificacion)
    conjunto_de_instancias_vectorizadas_normalizadas = normalizar(conjunto_de_instancias_vectorizadas)

    for i in range(50):
        #WARNING: poner funcion de activiacion=identidad hace que diverja todo al chori
        capa_1 = CapaInterna(cantidad_neuronas=10, funcion_activacion=tanh)
        capa_2 = CapaInterna(cantidad_neuronas=(i+1), funcion_activacion=tanh)
        capa_3 = CapaInterna(cantidad_neuronas=(i+1)*10, funcion_activacion=tanh)
        capa_5 = CapaSalida(cantidad_neuronas=1, funcion_activacion=tanh)
        capas = [capa_1,capa_2,capa_5]
        perceptron_multicapa = PerceptronMulticapa(capas)
        mejor_norma = perceptron_multicapa.entrenar(conjunto_de_instancias_vectorizadas_normalizadas, clasificaciones)
        #print "Entrenamiento con %d, Norma: %f neuronas" % (i+1,mejor_norma)


    
