from funcion import SigmoideaLogistica, Identidad
from red_neuronal import PerceptronMulticapa, Capa

if __name__ == "__main__":
    identidad = Identidad()
    sigmoidea = SigmoideaLogistica(cte=1)

    capa_1 = Capa(cantidad_neuronas=3, funcion_activacion=identidad)
    capa_1.set_valores([1, 2, 3])
    capa_2 = Capa(cantidad_neuronas=2, funcion_activacion=sigmoidea)
    capa_3 = Capa(cantidad_neuronas=3, funcion_activacion=sigmoidea)
    capa_4 = Capa(cantidad_neuronas=2, funcion_activacion=sigmoidea)
    capa_5 = Capa(cantidad_neuronas=1, funcion_activacion=sigmoidea)

    capas = [capa_1, capa_2, capa_3, capa_4, capa_5]

    perceptron_multicapa = PerceptronMulticapa(capas)
    perceptron_multicapa.inicializar_pesos()

    print "MATRICES DE PESO"
    for indice_matriz in range(perceptron_multicapa.cantidad_de_matrices_de_pesos()):
        print perceptron_multicapa.matriz_de_pesos_numero(indice_matriz)
        print "----------------------------------"

    perceptron_multicapa.entrenar()

    print "VALORES DE CAPA"
    for indice_capa in range(perceptron_multicapa.cantidad_de_capas()):
        print perceptron_multicapa.capa_numero(indice_capa).valores()
        print "----------------------------------"
