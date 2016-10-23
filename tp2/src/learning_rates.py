import numpy as np

#Diferentes Learnings rates que se pueden utilizar
class Learning_rate_tipo_1:
    def __init__(self,learning_rate_inicial=0.7 ,learning_rate_proporcional=0.5, alfa=0.1):
        self.learning_rate_inicial = learning_rate_inicial
        self.learning_rate_proporcional = learning_rate_proporcional
        self.alfa = alfa
    def calcular(self,epocas):
        return self.learning_rate_inicial * (1 + epocas * self.learning_rate_proporcional) ** -(self.alfa)

class Learning_rate_tipo_2:
    def __init__(self,learning_rate_inicial=0.7 ,learning_rate_proporcional=0.5):
        self.learning_rate_inicial = learning_rate_inicial
        self.learning_rate_proporcional = learning_rate_proporcional
    def calcular(self,epocas):
        return self.learning_rate_inicial * np.exp(-epocas/self.learning_rate_proporcional)

class Learning_rate_tipo_3:
    def __init__(self,learning_rate_inicial=0.7 ,learning_rate_proporcional=0.5):
        self.learning_rate_inicial = learning_rate_inicial
        self.learning_rate_proporcional = learning_rate_proporcional
    def calcular(self,epocas):
        return self.learning_rate_inicial / (1 + epocas * self.learning_rate_proporcional * self.learning_rate_inicial)