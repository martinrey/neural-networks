import numpy as np
from red_neuronal import Red_hebbs,Red_mapeo_caracteristicas
from lector_de_instancias import LectorDeInstancias
from adapters import InstanciaCompania
from learning_rates import Learning_rate_tipo_1,Learning_rate_tipo_2,Learning_rate_tipo_3
import sys, getopt

def parsear_a_mlp(inputs, outputs):
    inputs_parseados = np.asarray(inputs)
    outputs_parseados = np.asarray(outputs)
    return inputs_parseados, outputs_parseados

def normalizar(instancias):
    cantidad_de_instancias = len(instancias)
    medias = np.mean(instancias, axis=0)
    varianza_muestral = np.var(instancias, axis=0)
    for i in range(cantidad_de_instancias):
        instancias[i] = (instancias[i] - medias) / np.sqrt(varianza_muestral)
    	instancias[i][varianza_muestral == 0] = 0
    return instancias


def cargar_problema_a_aprender(datos_csv, adapter):
    lector_de_instancias = LectorDeInstancias(archivo=datos_csv)
    conjunto_de_instancias_de_entrenamiento = lector_de_instancias.leer()
    conjunto_de_instancias_vectorizadas = []
    clasificaciones = []
    for instancia in conjunto_de_instancias_de_entrenamiento:
        instancia_vectorizada, clasificacion = adapter.adaptar_esto(instancia)
        conjunto_de_instancias_vectorizadas.append(instancia_vectorizada)
        clasificaciones.append(clasificacion)
    #conjunto_de_instancias_vectorizadas_normalizadas = normalizar(conjunto_de_instancias_vectorizadas)
    return parsear_a_mlp(conjunto_de_instancias_vectorizadas, clasificaciones)


def split(inputs, valor):
    proporcion = len(inputs) * valor
    return inputs[:int(proporcion)], inputs[int(proporcion):]

def generate_train_and_test_set(inputs,targets):
    proporcion = 0.2
    test_data, train_data = split(inputs, 0.2)
    test_targuets, train_targuets = split(targets,0.2)
    return (train_data,train_targuets) , (test_data,test_targuets)

def test_mapeo_1():
    #ejercicio 2 de la practica
    test = []
    test_res = []
    for i in range(1,3):
        for j in range(1,3):
            for cantidad_instancias in range(10):
                test.append([np.random.uniform(low=1+(i*10),high=2+(i*10)), np.random.uniform(low=1+(j*10),high=2+(j*10))])
                test_res.append([i+j])
    test = np.array(test)
    test_res = np.array(test_res)
    #Normalizar la entrada es necesario para obtener buenos resultados
    #No hacerlo reduce significativamente las capacidades de aprendisaje
    red_mapeo = Red_mapeo_caracteristicas(normalizar(test),10,10)
    red_mapeo.entrenar(Learning_rate_tipo_1())
    red_mapeo.testear(test,test_res)
    exit(0)

def print_flags():
    print 'python main.py numero_ejercicio'
    print "-i <inputfile> \t\t archivo de entrenamiento"
    print "-o <outputfile> \t Archivo donde guardar red"
    print "-n <net> \t\t Red a utilizar"
    print "-t <testing> \t\t Archivo contra el que testear"
    print "-g \t\t Graficar Resultados"

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
    testear_en_train_set = True

    if(len(sys.argv) <= 1):
        print_flags()
        sys.exit()
    argv = sys.argv[2:]
    option = int(sys.argv[1])
    try:
        opts, args = getopt.getopt(argv,"hi:o:n:t:g",["ifile=","ofile="])
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

    instancia_Compania_a_perceptron_adapter = InstanciaCompania()
    inputs, targets = cargar_problema_a_aprender(datos_csv=entrenamiento,
                                                 adapter=instancia_Compania_a_perceptron_adapter)
    train_set,testing_set = generate_train_and_test_set(inputs,targets)
    if option == 1:
        red_neuronal = Red_hebbs(train_set[0],3,'oja')
        learning_rate = 0.1
    if option == 2:
        red_neuronal = Red_hebbs(train_set[0],3,'sanger')
        learning_rate = 0.01
    if option == 3:
        red_neuronal = Red_mapeo_caracteristicas(train_set[0],10,10)
        learning_rate = Learning_rate_tipo_3()

    if cargar_red:
        red_neuronal.load_net(red_a_utilizar)
    else:
        red_neuronal.entrenar(learning_rate)
    if guardar_red:
        red_neuronal.save_net(output_net)
    if testear_en_test_set:
        red_neuronal.testear(testing_set[0],testing_set[1],title='Testeo Sobre Datos De Testeo',show_graphic=graficar_resultados)
    if testear_en_train_set:
        red_neuronal.testear(train_set[0],train_set[1],title='Testeo Sobre Datos De Entrenamiento',show_graphic=graficar_resultados)
    exit(0)