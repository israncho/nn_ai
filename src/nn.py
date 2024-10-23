'''Modulo para la implementación de nodos en una red neuronal.'''

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

class Node(ABC):
    '''Clase base abstracta para nodos en una red neuronal.'''

    def __init__(self, has_weights: bool = False):
        self.output: np.ndarray | float = None # type: ignore
        self.grad: np.ndarray | Tuple[np.ndarray, ...] | None = None
        self.has_weights = has_weights

    def __call__(self, x) -> np.ndarray | float:
        return self.forward(x)

    @abstractmethod
    def forward(self, x) -> np.ndarray | float:
        '''Ejecuta la operación de propagación hacia adelante
        para este nodo..'''

    @abstractmethod
    def backward(self, incoming_grad) -> np.ndarray | float | Tuple[np.ndarray, ...]:
        '''Calcula el gradiente durante la retropropagación
        para este nodo.'''
    
    
class Linear(Node):

    def __init__(self, input_size: int, output_size: int):
        super().__init__(has_weights = True)
        self.w = np.random.rand(input_size, output_size)
        self.b = np.random.rand(output_size)
        # salida de la capa anterior 
        self.h: Optional[np.ndarray] = None
    
    def forward(self, x) -> np.ndarray:
        self.h = x
        self.output = np.dot(self.w, x) + self.b
        return self.output
    
    def backward(self, incoming_grad) -> np.ndarray:
        '''todo'''
        pass


class ReLU(Node):
    
    def __init__(self):
        super().__init__()
        # preactivacion
        self.a: Optional[np.ndarray] = None

    def forward(self, x):
        self.a = x
        # funcion de maximo aplicado a cada entrada
        self.output = np.maximum(0, self.a)
        return self.output
    
    def backward(self, incoming_grad):
        # (self.a > 0) es matriz de booleanos (una mascara)
        self.grad = incoming_grad * (self.a > 0)
        return self.grad


class Tanh(Node):

    def __init__(self):
        super().__init__()
        self.a: Optional[np.ndarray] = None

    def forward(self, x):
        # evitando errores numericos pues np.tanh(19) = 1 !
        self.a = np.clip(x, -18.5, 18.5)
        self.output = np.tanh(self.a)
        return self.output 

    def backward(self, incoming_grad):
        self.grad = incoming_grad * (1 - self.output ** 2)
        return self.grad


class Softmax(Node):

    def __init__(self):
        super().__init__()
        self.a: Optional[np.ndarray] = None   

    def forward(self, x) -> np.ndarray:
        # evitando errores numericos pues np.exp(-746) = 0.0 y np.exp(710) = inf
        self.a = np.clip(x, -709, 709)
        exp_a = np.exp(self.a)
        # vector con las sumas aculadas de cada columna
        sum_column = np.sum(exp_a, axis=0)
        # division de cada elemento por su suma de columna correspondiente
        self.output = exp_a / sum_column
        return self.output
    
    def backward(self, incoming_grad) -> np.ndarray:
        self.grad = incoming_grad * (self.output * (1 - self.output))
        return self.grad