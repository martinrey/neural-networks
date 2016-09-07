from funcion import SigmoideaLogistica, Identidad
from red_neuronal import PerceptronMulticapa, Capa
import csv

if __name__ == "__main__":
    identidad = Identidad()
    sigmoidea = SigmoideaLogistica(cte=1)

    capa_1 = Capa(cantidad_neuronas=10, funcion_activacion=identidad)
    capa_2 = Capa(cantidad_neuronas=15, funcion_activacion=sigmoidea)
    capa_3 = Capa(cantidad_neuronas=15, funcion_activacion=sigmoidea)
    capa_4 = Capa(cantidad_neuronas=15, funcion_activacion=sigmoidea)
    capa_5 = Capa(cantidad_neuronas=1, funcion_activacion=sigmoidea)

    capas = [capa_1, capa_2, capa_3, capa_4, capa_5]

    perceptron_multicapa = PerceptronMulticapa(capas)
    perceptron_multicapa.inicializar_pesos()

    # print "MATRICES DE PESO"
    # for indice_matriz in range(perceptron_multicapa.cantidad_de_matrices_de_pesos()):
    #     print perceptron_multicapa.matriz_de_pesos_numero(indice_matriz)
    #     print "----------------------------------"
    #
    # perceptron_multicapa.entrenar()
    #
    # print "VALORES DE CAPA"
    # for indice_capa in range(perceptron_multicapa.cantidad_de_capas()):
    #     print perceptron_multicapa.capa_numero(indice_capa).valores()
    #     print "----------------------------------"


    with open('tp1_ej1_training.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            clasificacion = 1 if row[0] == 'M' else 0
            radio = row[1]
            textura = row[2]
            perimetro = row[3]
            area = row[4]
            suavidad = row[5]
            compacidad = row[6]
            concavidad = row[7]
            puntos_concavos = row[8]
            simetria = row[9]
            algo_mas = row[10]

            capa_1.set_valores(
                [radio, textura, perimetro, area, suavidad, compacidad, concavidad, puntos_concavos, simetria,
                 algo_mas])
            perceptron_multicapa.entrenar(clasificacion)
