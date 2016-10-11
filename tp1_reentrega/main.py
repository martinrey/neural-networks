from lector_de_instancias import LectorDeInstancias
from adapters import InstanciaCancerAPerceptronAdapter, InstanciaCargaEnergeticaAPerceptronAdapter
import numpy as np
import perceptron_multicapa


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

    return parsear_a_mlp(conjunto_de_instancias_vectorizadas_normalizadas, clasificaciones)


def split(inputs, valor):
    proporcion = len(inputs) * valor
    return inputs[:int(proporcion)], inputs[int(proporcion):]


if __name__ == "__main__":
    instancia_cancer_a_perceptron_adapter = InstanciaCancerAPerceptronAdapter()
    inputs, targets = cargar_problema_a_aprender(datos_csv='tp1_ej1_training.csv',
                                                 adapter=instancia_cancer_a_perceptron_adapter)
    for i in range(40):
        inputs_test, inputs_entrenamiento = split(inputs, 1.0/4 )
        targets_test, targets_entrenamiento = split(targets, 1.0/4 )
        q = perceptron_multicapa.PerceptronMulticapa(inputs_entrenamiento, targets_entrenamiento, 13)
        q.entrenar(inputs_entrenamiento, targets_entrenamiento, 0.02, 1500, "logistica", vervose=0)
        q.matriz_de_confusion(inputs_test, targets_test)


    # for i in range(25):
    #     instancia_carga_energetica_a_perceptron_adapter = InstanciaCargaEnergeticaAPerceptronAdapter()
    #     inputs, targets = cargar_problema_a_aprender(datos_csv='tp1_ej2_training.csv',
    #                                                  adapter=instancia_carga_energetica_a_perceptron_adapter)
    #     inputs_test, inputs_entrenamiento = split(inputs, 1.0/3 )
    #     targets_test, targets_entrenamiento = split(targets, 1.0/3 )
    #     q = perceptron_multicapa.PerceptronMulticapa(inputs_entrenamiento, targets_entrenamiento, (i+1)*200)
    #     q.entrenar(inputs_entrenamiento, targets_entrenamiento, 0.02, 5001, "lineal")
    #     q.matriz_de_confusion(inputs_test, targets_test)
