import csv


class LectorDeInstancias(object):
    def __init__(self, archivo):
        self._archivo = archivo

    def leer(self):
        with open(self._archivo, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            lineas = []
            for linea in reader:
                if linea == []:
                    break
                lineas.append(linea)
            return lineas
