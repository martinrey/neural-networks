from scipy import sparse

class InstanciaCompania(object):
    def adaptar_esto(self, instancia):
    	instancia_adaptada = []
        for elemento in instancia[1:]:
        	instancia_adaptada.append(float(elemento))
        clasificacion_adaptada = [float(instancia[0])]
        return [instancia_adaptada,clasificacion_adaptada]