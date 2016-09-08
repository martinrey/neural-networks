from funcion import SigmoideaLogistica, Identidad
from lector_de_instancias import LectorDeInstancias
from modelos.tumor import Tumor, TumorAPerceptronAdapter
from red_neuronal import PerceptronMulticapa, CapaInterna, CapaSalida

if __name__ == "__main__":
    identidad = Identidad()
    sigmoidea = SigmoideaLogistica(cte=1)

    capa_1 = CapaInterna(cantidad_neuronas=2, funcion_activacion=identidad)
    capa_2 = CapaInterna(cantidad_neuronas=2, funcion_activacion=sigmoidea)
    capa_3 = CapaInterna(cantidad_neuronas=2, funcion_activacion=sigmoidea)
    capa_4 = CapaInterna(cantidad_neuronas=2, funcion_activacion=sigmoidea)
    capa_5 = CapaSalida(cantidad_neuronas=1, funcion_activacion=sigmoidea)

    capas = [capa_1, capa_2, capa_3, capa_4, capa_5]

    perceptron_multicapa = PerceptronMulticapa(capas)
    perceptron_multicapa.inicializar_pesos()

    for indice_matriz in range(perceptron_multicapa.cantidad_de_matrices_de_pesos()):
        print "CAPA"
        print perceptron_multicapa.capa_numero(indice_matriz).valores()
        print "MATRIZ"
        print perceptron_multicapa.matriz_de_pesos_numero(indice_matriz)
    print "CAPA"
    print perceptron_multicapa.capa_numero(4).valores()

    lector_de_tumores = LectorDeInstancias(archivo='tp1_ej1_training.csv', clase_de_las_instancias=Tumor)
    conjunto_de_tumores_de_entrenamiento = lector_de_tumores.leer()
    conjunto_de_tumores_vectorizado = []
    clasificaciones = []
    for tumor in conjunto_de_tumores_de_entrenamiento:
        tumor_vectorizado, clasificacion = TumorAPerceptronAdapter().adaptar_esto(tumor)
        conjunto_de_tumores_vectorizado.append(tumor_vectorizado)
        clasificaciones.append(clasificacion)

    perceptron_multicapa.entrenar(conjunto_de_tumores_vectorizado, clasificaciones)


