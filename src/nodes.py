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

    def __call__(self, x) -> Union[np.ndarray, float]:
        return self.forward(x)

    @abstractmethod
    def forward(self, x) -> Union[np.ndarray, float]:
        '''Ejecuta la operación con la entrada proporcionada.'''


class PreActivation(OperationNode):
    '''Nodo que calcula el valor de pre-activación
    (suma ponderada más sesgo)'''

    def __init__(self, w: Weights, b: Bias):
        self.w = w
        self.b = b

    def forward(self, x):
        return np.dot(self.w(), x.T) + self.b()


class Sigmoid(OperationNode):
    '''Nodo que aplica la función de activación sigmoide
    a la salida de un nodo de operación anterior.'''

    def __init__(self, previous: OperationNode):
        self.previous = previous

    def forward(self, x):
        return 1 / (1 + np.exp(- self.previous(x)))
