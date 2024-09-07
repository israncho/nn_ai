from abc import ABC, abstractmethod
from typing import Union
import numpy as np

class DataNode(ABC):
    '''Clase abstracta para nodos de datos en
    la grafica computacional.'''

    def __call__(self) -> Union[np.ndarray, float]:
        return self.forward()

    @abstractmethod
    def forward(self) -> Union[np.ndarray, float]:
        '''Devuelve los datos del nodo.'''


class Weights(DataNode):
    '''Nodo que contiene pesos inicializados aleatoriamente.'''

    def __init__(self, dimension: int):
        self.w = np.random.uniform(0, 1, size=dimension)

    def forward(self):
        return self.w


class Bias(DataNode):
    '''Nodo que contiene un valor de sesgo inicializado aleatoriamente.'''

    def __init__(self):
        self.b = np.random.random()

    def forward(self):
        return self.b


class OperationNode(ABC):
    '''Clase abstracta para nodos que realizan calculos
    con datos de entrada y salidas de otros nodos.'''

    def __call__(self, *args) -> Union[np.ndarray, float]:
        return self.forward(*args)

    @abstractmethod
    def forward(self, *args) -> Union[np.ndarray, float]:
        '''Ejecuta la operación con la entrada proporcionada.'''


class PreActivation(OperationNode):
    '''Nodo que calcula el valor de pre-activación
    (suma ponderada más sesgo)'''

    def __init__(self, w: Weights, b: Bias):
        self.w = w
        self.b = b

    def forward(self, *args):
        x = args[0]
        return np.dot(self.w(), x.T) + self.b()


class Sigmoid(OperationNode):
    '''Nodo que aplica la función de activación sigmoide
    a la salida de un nodo de operación anterior.'''

    def __init__(self, previous: OperationNode):
        self.previous = previous

    def forward(self, *args):
        x = args[0]
        return 1 / (1 + np.exp(- self.previous(x)))


class BinCrossEntropy(OperationNode):
    '''Nodo que calcula la entropía cruzada binaria entre las
    predicciones del modelo y las etiquetas verdaderas.'''

    def __init__(self, previous: OperationNode):
        self.previous = previous

    def forward(self, *args):
        x = args[0]
        y = args[1]
        f_x = self.previous(x)
        return -(y * np.log(f_x) + (1 - y) * np.log(1 - f_x))


if __name__ == "__main__":
    logistic_reg = Sigmoid(PreActivation(Weights(2), Bias()))
    data_set = np.array([[1, 1], [2, 2], [3, 3], [2, 3]])
    data_set_labels = np.array([1, 1, 0, 0])
    print("dataset:\n", data_set)
    print("etiquetas:", data_set_labels)
    # regresion logistica para cada entrada
    # del dataset
    print("regresion del dataset: ", logistic_reg(data_set))
    # regresion logistica de un solo dato del dataset
    print("regresion de un solo dato: ", logistic_reg(data_set[0]))
    loss = BinCrossEntropy(logistic_reg)
    # error de cada entrada del dataset
    print("error: ", loss(data_set, data_set_labels))
