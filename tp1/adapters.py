class InstanciaAPerceptronAdapter(object):
    def adaptar_esto(self, instancia):
        return [float(instancia[1]), float(instancia[2]), float(instancia[3]), float(instancia[4]),
                float(instancia[5]), float(instancia[6]), float(instancia[7]), float(instancia[8]), float(instancia[9]),
                float(instancia[10])], (1.0 if 'M' == instancia[0] else 0.0)