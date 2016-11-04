from lector_de_instancias import LectorDeInstancias
from adapters import InstanciaCancerAPerceptronAdapter, InstanciaCargaEnergeticaAPerceptronAdapter
import numpy as np
import perceptron_multicapa
from funcion import SigmoideaLogistica, Identidad
import sys, getopt
import random


def normalizar(instancias):
    cantidad_de_instancias = len(instancias)
    medias = np.mean(instancias, axis=0)
    varianza_muestral = np.var(instancias, axis=0)
    for i in range(cantidad_de_instancias):
        instancias[i] = (instancias[i] - medias) / np.sqrt(varianza_muestral)
    return instancias


def parsear_a_mlp(inputs, outputs):
    inputs_parseados = np.asarray(inputs)
    outputs_parseados = np.asarray(outputs)
    return inputs_parseados, outputs_parseados


def cargar_problema_a_aprender(datos_csv, adapter):
    lector_de_instancias = LectorDeInstancias(archivo=datos_csv)
    conjunto_de_instancias_de_entrenamiento = lector_de_instancias.leer()
    conjunto_de_instancias_vectorizadas = []
    clasificaciones = []
    for instancia in conjunto_de_instancias_de_entrenamiento:
        instancia_vectorizada, clasificacion = adapter.adaptar_esto(instancia)
        conjunto_de_instancias_vectorizadas.append(instancia_vectorizada)
        clasificaciones.append(clasificacion)
    conjunto_de_instancias_vectorizadas_normalizadas = normalizar(conjunto_de_instancias_vectorizadas)

    random.seed(10)
    combined = list(zip(conjunto_de_instancias_vectorizadas_normalizadas, clasificaciones))
    random.shuffle(combined)
    conjunto_de_instancias_vectorizadas_normalizadas[:], clasificaciones[:] = zip(*combined)

    return parsear_a_mlp(conjunto_de_instancias_vectorizadas_normalizadas, clasificaciones)


def split(inputs, valor):
    proporcion = len(inputs) * valor
    return inputs[:int(proporcion)], inputs[int(proporcion):]


# if __name__ == "__main__":
#     instancia_cancer_a_perceptron_adapter = InstanciaCancerAPerceptronAdapter()
#     inputs, targets = cargar_problema_a_aprender(datos_csv='tp1_ej1_training.csv',
#                                                  adapter=instancia_cancer_a_perceptron_adapter)
#     # for i in range(40):
#     #     funciones=[SigmoideaLogistica(1),SigmoideaLogistica(1),SigmoideaLogistica(1)]
#     #     inputs_test, inputs_entrenamiento = split(inputs, 1.0/4 )
#     #     targets_test, targets_entrenamiento = split(targets, 1.0/4 )
#     #     q = perceptron_multicapa.PerceptronMulticapa(inputs_entrenamiento, targets_entrenamiento, 7,funciones)
#     #     q.entrenar( 0.02, 3500, "logistica", verbose=0)
#     #     q.matriz_de_confusion(inputs_test, targets_test)


#     instancia_carga_energetica_a_perceptron_adapter = InstanciaCargaEnergeticaAPerceptronAdapter()
#     inputs, targets = cargar_problema_a_aprender(datos_csv='tp1_ej2_training.csv',
#                                                      adapter=instancia_carga_energetica_a_perceptron_adapter)
#     inputs_test, inputs_entrenamiento = split(inputs, 1.0/3 )
#     targets_test, targets_entrenamiento = split(targets, 1.0/3 )
#     for i in range(25):
#         funciones=[SigmoideaLogistica(1),SigmoideaLogistica(1),Identidad()]
#         q = perceptron_multicapa.PerceptronMulticapa(inputs_entrenamiento, targets_entrenamiento, 30,funciones)
#         q.entrenar(0.02, 50001, "lineal", verbose=1)
#         q.comparar_resultdos(inputs_test, targets_test,"lineal")

def print_flags():
    print 'python main.py numero_ejercicio'
    print "-i <inputfile> \t\t archivo de entrenamiento"
    print "-o <outputfile> \t Archivo donde guardar red"
    print "-n <net> \t\t Red a utilizar"
    print "-t <testing> \t\t Archivo contra el que testear"
    print "-g \t\t Graficar Resultados"
    print "-e \t\t Cantidad de Epocas"
    print "-s \t\t Silent"
    print "-l \t\t learning rate"
    print "-t \t\t Testear en training set"
    print "-c \t\t Cantidad De Neuronas"
    print "-w \t\t Testear En datos de Entrenamiento (deshabilita testeo en datos de testeo)"


if __name__ == "__main__":
    uso_input = False
    guardar_red = False
    cargar_red = False
    testear_resultados = False
    entrenamiento ='tp2_training_dataset.csv'
    output_net=''
    red_a_utilizar=''
    testing = ''
    graficar_resultados = False
    testear_en_test_set = True
    testear_en_train_set = False
    default_epocas = True
    default_learning_rate = True
    silent = False
    cantidad_de_neuronas_default = True

    if(len(sys.argv) <= 1):
        print_flags()
        sys.exit()
    argv = sys.argv[2:]
    option = int(sys.argv[1])
    try:
        opts, args = getopt.getopt(argv,"hi:o:n:t:e:gsl:wc:",["ifile=","ofile="])
    except getopt.GetoptError:
        print_flags()
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print_flags()
            sys.exit()
        elif opt in ("-i", "--inputfile"):
            uso_input = True
            entrenamiento = arg
        elif opt in ("-o", "--outputfile"):
            guardar_red = True
            output_net = arg
        elif opt in ("-n", "--net"):
            cargar_red = True
            red_a_utilizar = arg
        elif opt in ("-t", "--testing"):
            testear_resultados = True
            testing = arg
        elif opt in ("-g", "--graphix"):
            graficar_resultados = True
        elif opt in ("-e", "--epoc"):
            default_epocas = False
            cantidad_iteraciones = int(arg)
        elif opt in ("-s", "--silent"):
            silent = True
        elif opt in ("-l", "--silent"):
            default_learning_rate = False
            learning_rate = float(arg)
        elif opt in ("-w", "--training"):
                testear_en_test_set = False
                testear_en_train_set = True
        elif opt in ("-c", "--training"):
            cantidad_de_neuronas = int(arg)
            cantidad_de_neuronas_default = False

    if option == 1:
        instancia_cancer_a_perceptron_adapter = InstanciaCancerAPerceptronAdapter()
        inputs, targets = cargar_problema_a_aprender(datos_csv='tp1_ej1_training.csv',adapter=instancia_cancer_a_perceptron_adapter)
        funciones=[SigmoideaLogistica(1),SigmoideaLogistica(1),SigmoideaLogistica(1)]
        inputs_test, inputs_entrenamiento = split(inputs, 1.0/4 )
        targets_test, targets_entrenamiento = split(targets, 1.0/4 )
        tipo_func = "logistica"
        if default_epocas:
            cantidad_iteraciones = 900
        if default_learning_rate:
            learning_rate = 0.02
        if cantidad_de_neuronas_default:
            cantidad_de_neuronas = 3

    if option == 2:
        instancia_carga_energetica_a_perceptron_adapter = InstanciaCargaEnergeticaAPerceptronAdapter()
        inputs, targets = cargar_problema_a_aprender(datos_csv='tp1_ej2_training.csv',adapter=instancia_carga_energetica_a_perceptron_adapter)
        inputs_test, inputs_entrenamiento = split(inputs, 1.0/3 )
        targets_test, targets_entrenamiento = split(targets, 1.0/3 )
        funciones=[SigmoideaLogistica(1),SigmoideaLogistica(1),Identidad()]
        if default_epocas:
            cantidad_iteraciones = 8000
        tipo_func = "lineal"
        if default_learning_rate:
            learning_rate = 0.05
        if cantidad_de_neuronas_default:
            cantidad_de_neuronas = 8
    
    red_neuronal = perceptron_multicapa.PerceptronMulticapa(inputs_entrenamiento, targets_entrenamiento, cantidad_de_neuronas,funciones)
    if cargar_red:
        red_neuronal.load_net(red_a_utilizar)
    else:
        red_neuronal.entrenar(learning_rate, cantidad_iteraciones, tipo_func, verbose=not silent)
    if guardar_red:
        red_neuronal.save_net(output_net)
    if testear_en_test_set:
        if option == 1:
            red_neuronal.matriz_de_confusion(inputs_test, targets_test)
        if option == 2:
            red_neuronal.comparar_resultdos(inputs_test, targets_test,"lineal")
    if testear_en_train_set:
        if option == 1:
            red_neuronal.matriz_de_confusion(inputs_entrenamiento, targets_entrenamiento)
        if option == 2:
            red_neuronal.comparar_resultdos(inputs_entrenamiento, targets_entrenamiento,"lineal")
    exit(0)