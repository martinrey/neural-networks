from funcion import SigmoideaLogistica, Identidad
from lector_de_instancias import LectorDeInstancias
from red_neuronal import PerceptronMulticapa, CapaInterna, CapaSalida
from adapters import InstanciaAPerceptronAdapter

if __name__ == "__main__":
    identidad = Identidad()
    sigmoidea = SigmoideaLogistica(cte=1)

    capa_1 = CapaInterna(cantidad_neuronas=10, funcion_activacion=identidad)
    capa_2 = CapaInterna(cantidad_neuronas=20, funcion_activacion=identidad)
    capa_3 = CapaInterna(cantidad_neuronas=20, funcion_activacion=identidad)
    capa_4 = CapaInterna(cantidad_neuronas=20, funcion_activacion=identidad)
    capa_5 = CapaSalida(cantidad_neuronas=1, funcion_activacion=identidad)

    capas = [capa_1, capa_2, capa_3, capa_4, capa_5]

    perceptron_multicapa = PerceptronMulticapa(capas)
    perceptron_multicapa.inicializar_pesos()

    lector_de_instancias = LectorDeInstancias(archivo='tp1_ej1_training.csv')
    conjunto_de_instancias_de_entrenamiento = lector_de_instancias.leer()
    conjunto_de_instancias_vectorizadas = []
    clasificaciones = []
    for instancia in conjunto_de_instancias_de_entrenamiento:
        instancia_vectorizada, clasificacion = InstanciaAPerceptronAdapter().adaptar_esto(instancia)
        conjunto_de_instancias_vectorizadas.append(instancia_vectorizada)
        clasificaciones.append(clasificacion)

    perceptron_multicapa.entrenar(conjunto_de_instancias_vectorizadas, clasificaciones)


