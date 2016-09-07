import numpy as np

def funcionActivacion( array ):
	#TODO: recorro los valores del array y los seteo en 1 o -1
	#ver pag 136 simon o. hakin
	vfunc = np.vectorize(logFunc)
	return vfunc(array)


cteLog = 1
def logFunc( x ):
	return 1/ (1.0 + np.exp(cteLog * x)) 

def derivadaLogFunc( x ):
	return cteLog*logFunc(x)*(1- logFunc(x))

def derivadaFuncionActivacion( array ):
	#TODO: derivar y devolver valor, bla bla
	vfunc = np.vectorize(derivadaLogFunc)
	return vfunc(array)

if __name__ == "__main__":
	#Valores para ejemplo
	layers = 3
	cantNeuronasPorLayer = [4, 2,4]
	matricesPesos = []
	learningRate = 1
	#Armo las n-1 matrices de pesos
	#TODO: Agregar valores aleatorios
	entrada = np.zeros( (1 ,cantNeuronasPorLayer[0]) )
	for i in range(layers-1):
		matricesPesos.append(np.zeros((cantNeuronasPorLayer[i], cantNeuronasPorLayer[i+1])))

	#foward-propagation
	capaNeuronal = []
	capaNeuronal.append(entrada)
	for i in range(layers-1):
		nuevacapa = funcionActivacion(np.dot(capaNeuronal[i],matricesPesos[i]))
		capaNeuronal.append(nuevacapa)

	salidaObtenida = capaNeuronal[layers-1]
	salidaEsperada = capaNeuronal[layers-1]

	#backwards-propagation

	#paso 1, calculo el gradiente local en la ultima capa
	#gradiente local = errorUltimaCapa * derivada de la funcion de activacion evaluada en el vector
	#(ver simon o. haykin, neural networks and learning machines, pag 134)

	errorUltimaCapa = np.subtract(salidaEsperada, salidaObtenida)
	gradienteLocal = []
	gradienteLocal.append(np.multiply(errorUltimaCapa, derivadaFuncionActivacion(salidaObtenida)))

	#paso 2. calculo el gradiente local para las capas ocultas
	#gradiente local = derivada de la funcion en el vector de activacion por (la sumatoria del gradiente local de la capa a la derecha multiplicado por los pesos sinapticos asociados entre estas dos capas)
	#(ver simon o. haykin, neural networks and learning machines, pag 134)

	for i in range(layers-1):
		print i
		derivada = derivadaFuncionActivacion(capaNeuronal[layers-1-i]) 
		print derivada
		print gradienteLocal[i]
		print matricesPesos[layers-2-i]
		sumatoria = np.sum(np.dot( gradienteLocal[i], np.transpose(matricesPesos[layers-2-i]) ), axis=1)
		gradiente = np.multiply(derivada, sumatoria)
		print "gradiente"
		print gradiente
		gradienteLocal.append(gradiente) 


