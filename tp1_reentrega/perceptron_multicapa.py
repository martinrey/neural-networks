import numpy as np


class PerceptronMulticapa:
    def __init__(self, inputs, targets, nhidden, beta=1, momentum=0.9):
        self.cantidad_neuronas_entrada = np.shape(inputs)[1]
        self.cantidad_neuronas_salida = np.shape(targets)[1]
        self.cantidad_instancias_dataset = np.shape(inputs)[0]
        self.cantidad_neuronas_capa_oculta = nhidden

        self.beta = beta
        self.momentum = momentum

        self.weights1 = (np.random.rand(self.cantidad_neuronas_entrada + 1, self.cantidad_neuronas_capa_oculta) - 0.5) * 2 / np.sqrt(self.cantidad_neuronas_entrada)
        self.weights2 = (np.random.rand(self.cantidad_neuronas_capa_oculta + 1, self.cantidad_neuronas_salida) - 0.5) * 2 / np.sqrt(self.cantidad_neuronas_capa_oculta)

    def entrenar(self, inputs, targets, eta, niterations, tipo_funcion):
        inputs = np.concatenate((inputs, -np.ones((self.cantidad_instancias_dataset, 1))), axis=1)
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(niterations):
            outputs = self.forwarding(inputs, tipo_funcion)

            error = 0.5 * np.sum((outputs - targets) ** 2)
            if (np.mod(n, 1000) == 0):
                print "Iteration: ", n, " Error: ", error

            if tipo_funcion == 'lineal':
                deltao = (outputs - targets) / self.cantidad_instancias_dataset
            elif tipo_funcion == 'logistica':
                deltao = self.beta * (outputs - targets) * outputs * (1.0 - outputs)

            deltah = self.hidden * self.beta * (1.0 - self.hidden) * (np.dot(deltao, np.transpose(self.weights2)))

            updatew1 = eta * (np.dot(np.transpose(inputs), deltah[:, :-1])) + self.momentum * updatew1
            updatew2 = eta * (np.dot(np.transpose(self.hidden), deltao)) + self.momentum * updatew2

            self.weights1 -= updatew1
            self.weights2 -= updatew2

    def forwarding(self, inputs, tipo_funcion):
        self.hidden = np.dot(inputs, self.weights1)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate((self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        outputs = np.dot(self.hidden, self.weights2)

        if tipo_funcion == 'lineal':
            return outputs
        elif tipo_funcion == 'logistica':
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))

    def matriz_de_confusion(self, inputs, targets):
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        outputs = self.forwarding(inputs, "logistica")
        nclasses = 2
        outputs = np.where(outputs > 0.5, 1, 0)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        print "Confusion matrix is:"
        print cm
        print "Percentage Correct: ", np.trace(cm) / np.sum(cm) * 100
