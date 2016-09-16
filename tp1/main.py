from funcion import SigmoideaLogistica, Identidad, Tanh
from lector_de_instancias import LectorDeInstancias
from red_neuronal import PerceptronMulticapa, CapaInterna, CapaSalida
from adapters import InstanciaAPerceptronAdapter

def normalizar(instancias):
    medias = []
    varianza_muestral = []
    cantidad_de_instancias = len(instancias)
    cantidad_de_atributos = len(instancias[0])
    for i in range(cantidad_de_atributos):
        medias.append(0)
        varianza_muestral.append(0)
    for i in range(cantidad_de_instancias):
        for j in range(cantidad_de_atributos):
            medias[j] += instancias[i][j]
    for i in range(cantidad_de_atributos):
        medias[i] = medias[i]/(1.0*cantidad_de_instancias)
    for i in range(cantidad_de_atributos):
        for j in range(cantidad_de_atributos):
            varianza_muestral[j] += (instancias[i][j] - medias[j]) ** 2
    for j in range(cantidad_de_atributos):
        varianza_muestral[j] = varianza_muestral[j]/(cantidad_de_instancias-1)

    for i in range(cantidad_de_instancias):
        for j in range(cantidad_de_atributos):
            instancias[i][j] = (instancias[i][j]-medias[j])/varianza_muestral[j]
    return instancias

if __name__ == "__main__":
    tanh = Tanh()
    identidad = Identidad()
    sigmoidea = SigmoideaLogistica(cte=1)

    #WARNING: poner funcion de activiacion=identidad hace que diverja todo al chori
    capa_1 = CapaInterna(cantidad_neuronas=10, funcion_activacion=sigmoidea)
    capa_2 = CapaInterna(cantidad_neuronas=200, funcion_activacion=sigmoidea)
    capa_3 = CapaInterna(cantidad_neuronas=100, funcion_activacion=sigmoidea)
    capa_4 = CapaInterna(cantidad_neuronas=20, funcion_activacion=sigmoidea)
    capa_5 = CapaSalida(cantidad_neuronas=1, funcion_activacion=sigmoidea)

    capas = [capa_1,capa_2, capa_5]

    perceptron_multicapa = PerceptronMulticapa(capas)

    lector_de_instancias = LectorDeInstancias(archivo='tp1_ej1_training.csv')
    conjunto_de_instancias_de_entrenamiento = lector_de_instancias.leer()
    conjunto_de_instancias_vectorizadas = []
    clasificaciones = []
    for instancia in conjunto_de_instancias_de_entrenamiento:
        instancia_vectorizada, clasificacion = InstanciaAPerceptronAdapter().adaptar_esto(instancia)
        conjunto_de_instancias_vectorizadas.append(instancia_vectorizada)
        clasificaciones.append(clasificacion)
    conjunto_de_instancias_vectorizadas_normalizadas = normalizar(conjunto_de_instancias_vectorizadas)
    perceptron_multicapa.inicializar_pesos(len(conjunto_de_instancias_vectorizadas))
    perceptron_multicapa.entrenar(conjunto_de_instancias_vectorizadas_normalizadas, clasificaciones)


    
