from funcion import SigmoideaLogistica, Identidad, Tanh
from lector_de_instancias import LectorDeInstancias
from red_neuronal import PerceptronMulticapa, CapaInterna, CapaSalida
from adapters import InstanciaAPerceptronAdapterCalor, InstanciaAPerceptronAdapterCancer
import numpy as np
import sys, getopt


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
        #else:
            #print "outlier encontrado!"
            #print np.logical_or(np.less(inst, cota_inferior), np.greater(inst, cota_superior))
    cantidad_de_instancias = len(resutaldo)
    medias = np.mean(resutaldo,axis=0)
    varianza_muestral = np.var(resutaldo,axis=0)
    cantidad_de_atributos = len(medias)
    for i in range(cantidad_de_instancias):
        resutaldo[i] = (resutaldo[i]-medias)/varianza_muestral
    return resutaldo

def cargando_red(red,testing):
    tanh = Tanh()
    identidad = Identidad()
    sigmoidea = SigmoideaLogistica(cte=1)
    capa_1 = CapaInterna(cantidad_neuronas=10, funcion_activacion=identidad)
    capa_2 = CapaInterna(cantidad_neuronas=70, funcion_activacion=tanh)
    capa_5 = CapaSalida(cantidad_neuronas=1, funcion_activacion=tanh)
    capas = [capa_1,capa_2,capa_5]
    perceptron_multicapa = PerceptronMulticapa(capas)
    mejor_norma = perceptron_multicapa.load_net(red)
    perceptron_multicapa.testear_resultados(testing, verbose=1)
    sys.exit()

if __name__ == "__main__":

    uso_input = False
    guardar_red = False
    cargar_red = False
    testear_resultados = False

    if(len(sys.argv) <= 1):
        print 'test.py numero_ejercicio'
        print "-i <inputfile> \t\t archivo de entrenamiento"
        print "-o <outputfile> \t Archivo donde guardar red"
        print "-n <net> \t\t Red a utilizar"
        print "-t <testing> \t\t Archivo contra el que testear"
        sys.exit()

    argv = sys.argv[2:]
    option = int(sys.argv[1])

    try:
        opts, args = getopt.getopt(argv,"hi:o:n:t:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'test.py numero_ejercicio'
        print "-i <inputfile> \t\t archivo de entrenamiento"
        print "-o <outputfile> \t Archivo donde guardar red"
        print "-n <net> \t\t Red a utilizar"
        print "-t <testing> \t\t Archivo contra el que testear"
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py numero_ejercicio'
            print "-i <inputfile> \t\t archivo de entrenamiento"
            print "-o <outputfile> \t Archivo donde guardar red"
            print "-n <net> \t\t Red a utilizar"
            print "-t <testing> \t\t Archivo contra el que testear"
            sys.exit()
        elif opt in ("-i", "--inputfile"):
            uso_input = True
            inputfile = arg
        elif opt in ("-o", "--outputfile"):
            guardar_red = True
            outputfile = arg
        elif opt in ("-n", "--net"):
            cargar_red = True
            red_a_utilizar = arg
        elif opt in ("-t", "--testing"):
            testear_resultados = True
            testing = arg

    if(cargar_red):
        cargando_red(red_a_utilizar,testing)

    if(option == 1):
        if(uso_input):
            lector_de_instancias = LectorDeInstancias(archivo=inputfile)
        else:
            lector_de_instancias = LectorDeInstancias(archivo='tp1_ej1_training.csv')
        adaptador = InstanciaAPerceptronAdapterCancer()
        output = 1
    if(option == 2):
        if(uso_input):
            lector_de_instancias = LectorDeInstancias(archivo=inputfile)
        else:
            lector_de_instancias = LectorDeInstancias(archivo='tp1_ej2_training.csv')
        adaptador = InstanciaAPerceptronAdapterCalor()
        output = 2
    conjunto_de_instancias_de_entrenamiento = lector_de_instancias.leer()
    conjunto_de_instancias_vectorizadas = []
    clasificaciones = []
    for instancia in conjunto_de_instancias_de_entrenamiento:
        instancia_vectorizada, clasificacion = adaptador.adaptar_esto(instancia)
        conjunto_de_instancias_vectorizadas.append(instancia_vectorizada)
        clasificaciones.append(np.array(clasificacion))
    conjunto_de_instancias_vectorizadas_normalizadas = normalizar(conjunto_de_instancias_vectorizadas)

    tanh = Tanh()
    identidad = Identidad()
    sigmoidea = SigmoideaLogistica(cte=1)
    for i in range(1):
        capa_1 = CapaInterna(cantidad_neuronas=len(conjunto_de_instancias_vectorizadas_normalizadas[0]), funcion_activacion=identidad)
        capa_2 = CapaInterna(cantidad_neuronas=70, funcion_activacion=tanh)
        capa_3 = CapaInterna(cantidad_neuronas=10,funcion_activacion=tanh)
        capa_4 = CapaInterna(cantidad_neuronas=8, funcion_activacion=tanh)
        capa_5 = CapaSalida(cantidad_neuronas=output, funcion_activacion=tanh)
        capas = [capa_1,capa_2,capa_5]
        perceptron_multicapa = PerceptronMulticapa(capas)
        mejor_norma = perceptron_multicapa.entrenar(conjunto_de_instancias_vectorizadas_normalizadas, np.array(clasificaciones),verbose=1,coeficiente_aprendisaje=1.0/46)
        if(testear_resultados):
            perceptron_multicapa.testear_resultados(testing, verbose=1)
        if(guardar_red):
            perceptron_multicapa.save_net(outputfile)
        #print "%d %f" % (1.0/(i+1),mejor_norma)


