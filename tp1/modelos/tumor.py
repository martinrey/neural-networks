class Tumor(object):
    def __init__(self, clasificacion, radio, textura, perimetro, area, suavidad, compacidad, concavidad,
                 puntos_concavos, simetria, algo_mas):
        self.clasificacion = self._obtener_clasificacion_a_partir_de(clasificacion)
        self.radio = radio
        self.textura = textura
        self.perimetro = perimetro
        self.area = area
        self.suavidad = suavidad
        self.compacidad = compacidad
        self.concavidad = concavidad
        self.puntos_concavos = puntos_concavos
        self.simetria = simetria
        self.algo_mas = algo_mas

    def _obtener_clasificacion_a_partir_de(self, clasificacion):
        return 1 if 'M' == clasificacion else 0


class TumorAPerceptronAdapter(object):
    def adaptar_esto(self, tumor):
        return [float(tumor.radio), float(tumor.textura), float(tumor.perimetro), float(tumor.area),
                float(tumor.suavidad), float(tumor.compacidad),
                float(tumor.concavidad), float(tumor.puntos_concavos), float(tumor.simetria),
                float(tumor.algo_mas)], float(tumor.clasificacion)
