import numpy as np


class PerceptronMulticapa:
    def __init__(self, inputs, targets, nhidden,funciones, beta=1, momentum=0.9):
        self.cantidad_neuronas_entrada = np.shape(inputs)[1]
        self.cantidad_neuronas_salida = np.shape(targets)[1]
        self.cantidad_instancias_dataset = np.shape(inputs)[0]
        self.cantidad_neuronas_capa_oculta = nhidden

        self.inputs_entrenamiento = inputs
        self.targets_entrenamiento = targets
        #self.inputs_entrenamiento = np.concatenate((self.inputs_entrenamiento, -np.ones((self.cantidad_instancias_dataset, 1))), axis=1)
        self.funciones = funciones

        self.beta = beta
        self.momentum = momentum

        self.weights1 = np.random.normal(size=(self.cantidad_neuronas_entrada + 1, self.cantidad_neuronas_capa_oculta),scale= 1/np.sqrt(self.cantidad_neuronas_entrada))
        self.weights2 = np.random.normal(size=(self.cantidad_neuronas_capa_oculta + 1, self.cantidad_neuronas_salida),scale= 1/np.sqrt(self.cantidad_neuronas_capa_oculta))
        self.weights = [self.weights1,self.weights2]

    def entrenar(self, eta, niterations, tipo_funcion,verbose=0):
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
        for n in range(niterations):
            outputs = self.forwarding(self.inputs_entrenamiento, tipo_funcion)

            error = 0.5 * np.sum((outputs - self.targets_entrenamiento) ** 2)
            if (np.mod(n, 1000) == 0):
                if verbose:
                    print "Iteration: ", n, " Error: ", error

            if tipo_funcion == 'lineal':
                deltao = self.funciones[-1].derivar_y_evaluar_en(self.capas[-1])*(outputs - self.targets_entrenamiento)/np.linalg.norm(error)
            elif tipo_funcion == 'logistica':
                deltao = self.funciones[-1].derivar_y_evaluar_en(self.capas[-1])*(outputs - self.targets_entrenamiento)

            deltah = self.capas[1] * self.beta * (1.0 - self.capas[1]) * (np.dot(deltao, np.transpose(self.weights2)))
            updatew1 = eta * (np.dot(np.transpose(self.capas[0]), deltah[:, :-1])) + self.momentum * updatew1
            updatew2 = eta * (np.dot(np.transpose(self.capas[1]), deltao)) + self.momentum * updatew2

            self.weights1 -= updatew1
            self.weights2 -= updatew2

    def forwarding(self, inputs, tipo_funcion):
        self.capas = [inputs]
        #podria extenderse a mas capas, pero no se si es tan importante.
        for i in range(2):
            self.capas[i] = self.funciones[i].evaluar_en(self.capas[i])
            self.capas[i] = np.concatenate((self.capas[i], -np.ones((np.shape(self.capas[i])[0], 1))), axis=1)
            self.capas.append(np.dot(self.capas[i], self.weights[i]))
        outputs = self.capas[-1]
        return self.funciones[-1].evaluar_en(outputs)

    def matriz_de_confusion(self, inputs, targets):
        outputs = self.forwarding(inputs, "logistica")
        nclasses = 2
        outputs = np.where(outputs > 0.5, 1, 0)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        #print "Confusion matrix is:"
        print cm[0,0],cm[0,1],cm[1,0],cm[1,1]
        #print "Percentage Correct: ", np.trace(cm) / np.sum(cm) * 100

    def comparar_resultdos(self,inputs, targets,tipo_funcion):
        outputs = self.forwarding(inputs, tipo_funcion)
        error = 0.5 * np.sum((outputs - targets) ** 2)
        print "Norma Del Error:" , error