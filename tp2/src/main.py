import numpy as np
from red_neuronal import Red_hebbs,Red_mapeo_caracteristicas
from lector_de_instancias import LectorDeInstancias
from adapters import InstanciaCompania
from learning_rates import Learning_rate_tipo_1,Learning_rate_tipo_2,Learning_rate_tipo_3

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
    conjunto_de_instancias_vectorizadas_normalizadas = normalizar(conjunto_de_instancias_vectorizadas)
    return parsear_a_mlp(conjunto_de_instancias_vectorizadas, clasificaciones)


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

def correr_hebbs():
    instancia_Compania_a_perceptron_adapter = InstanciaCompania()
    inputs, targets = cargar_problema_a_aprender(datos_csv='tp2_training_dataset.csv',
                                                 adapter=instancia_Compania_a_perceptron_adapter)

    red_hebbs = Red_hebbs(inputs,3,'sanjer')
    red_hebbs.entrenar(0.01)
    red_hebbs.testear(inputs,targets)
    exit(0)
    
def correr_mapeo():
    instancia_Compania_a_perceptron_adapter = InstanciaCompania()
    inputs, targets = cargar_problema_a_aprender(datos_csv='tp2_training_dataset.csv',
                                                 adapter=instancia_Compania_a_perceptron_adapter)
    red_mapeo = Red_mapeo_caracteristicas(inputs,40,40)
    red_mapeo.entrenar(Learning_rate_tipo_3())
    red_mapeo.save_net()
    red_mapeo.testear(inputs,targets)
    exit(0)

if __name__ == "__main__":
    correr_mapeo()


