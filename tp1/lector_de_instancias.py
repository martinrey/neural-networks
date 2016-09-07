import csv


class LectorDeInstancias(object):
    def __init__(self, archivo, clase_de_las_instancias):
        self._archivo = archivo
        self._clase_interprete = clase_de_las_instancias

    def leer(self):
        with open(self._archivo, 'rt', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            instancias = []
            for linea in reader:
                if linea == []:
                    break
                instancia = self._clase_interprete(*linea)
                instancias.append(instancia)
            return instancias
