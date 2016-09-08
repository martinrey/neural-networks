from funcion import SigmoideaLogistica, Identidad
from lector_de_instancias import LectorDeInstancias
from modelos.tumor import Tumor, TumorAPerceptronAdapter
from red_neuronal import PerceptronMulticapa, Capa

if __name__ == "__main__":
    identidad = Identidad()
    sigmoidea = SigmoideaLogistica(cte=1)

    capa_1 = Capa(cantidad_neuronas=10, funcion_activacion=identidad,hidden=1)
    print capa_1.cantidad_neuronas()
    capa_2 = Capa(cantidad_neuronas=15, funcion_activacion=sigmoidea,hidden=1)
    capa_3 = Capa(cantidad_neuronas=19, funcion_activacion=sigmoidea,hidden=1)
    capa_4 = Capa(cantidad_neuronas=22, funcion_activacion=sigmoidea,hidden=1)
    capa_5 = Capa(cantidad_neuronas=1, funcion_activacion=sigmoidea,hidden=0)

    capas = [capa_1, capa_2, capa_3, capa_4, capa_5]
    #capas = [capa_1,capa_5]

    perceptron_multicapa = PerceptronMulticapa(capas)
    perceptron_multicapa.inicializar_pesos()

    lector_de_tumores = LectorDeInstancias(archivo='tp1_ej1_training.csv', clase_de_las_instancias=Tumor)
    conjunto_de_tumores_de_entrenamiento = lector_de_tumores.leer()
    conjunto_de_tumores_vectorizado = []
    clasificaciones = []
    for tumor in conjunto_de_tumores_de_entrenamiento:
        tumor_vectorizado, clasificacion = TumorAPerceptronAdapter().adaptar_esto(tumor)
        conjunto_de_tumores_vectorizado.append(tumor_vectorizado)
        clasificaciones.append(clasificacion)

    perceptron_multicapa.entrenar(conjunto_de_tumores_vectorizado, clasificaciones)


