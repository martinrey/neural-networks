import numpy as np
from red_neuronal import Red_hebbs,Red_mapeo_caracteristicas
from lector_de_instancias import LectorDeInstancias
from adapters import InstanciaCompania


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
    #conjunto_de_instancias_vectorizadas_normalizadas = normalizar(conjunto_de_instancias_vectorizadas)
    return parsear_a_mlp(conjunto_de_instancias_vectorizadas, clasificaciones)


def test_mapeo():
    test = np.array([[1,1],[3000,-1000]])
    test_res = np.array([])
    red_mapeo = Red_mapeo_caracteristicas(test,test_res,10,10)
    red_mapeo.entrenar(0.01)
    red_mapeo.testear(test)
    exit(0)

def correr_hebbs():
    instancia_Compania_a_perceptron_adapter = InstanciaCompania()
    inputs, targets = cargar_problema_a_aprender(datos_csv='tp2_training_dataset.csv',
                                                 adapter=instancia_Compania_a_perceptron_adapter)

    red_hebbs = Red_hebbs(inputs,targets,2)
    red_hebbs.entrenar(0.0001)
    red_hebbs.testear(inputs)
    exit(0)
    
def correr_mapeo():
    red_mapeo = Red_mapeo_caracteristicas(inputs,targets,40,40)
    red_mapeo.entrenar(0.01)
    red_mapeo.testear(linea)
    exit(0)

if __name__ == "__main__":
    test_mapeo()


