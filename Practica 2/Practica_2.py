import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    if x < 0:
        return 0
    else:
        return 1

def perceptron_simple(dataset, w):
    errors = []
    error = 1
    epsilon = 0.1
    epoca = 0
    max_epocas = 1000
    coeficiente_aprendizaje = 0.01

    while(error > epsilon and epoca < max_epocas):
        error = 0
        for x, expected in dataset:
            result = np.dot(w, x)
            E = expected - step_function(result)
            w += coeficiente_aprendizaje * E * x
            error += np.power(E, 2)
            
            # Plot separating hyperplane
            p = (-w[1],w[0]); m = p[0] / p[1]; b = w[2]
            xx = np.linspace(-1,2,100); y = [b + m * i for i in xx]
            plt.plot(xx, y, 'r--', [0,0,1,1],[0,1,0,1], 'bs')
        epoca += 1
        errors.append(error)
    for x, _ in dataset:
        result = np.dot(x, w)
        print("{}: {} -> {}".format(x[:2], result, step_function(result)))
    return

# APRENDER FUNCION OR
dataset = [
    (np.array([0,0,-1]), 0),
    (np.array([0,1,-1]), 1),
    (np.array([1,0,-1]), 1),
    (np.array([1,1,-1]), 1),
]
w = np.random.rand(3)

perceptron_simple(dataset=dataset, w=w)
