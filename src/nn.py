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

    def __call__(self, *x) -> np.ndarray:
        return self.forward(*x)

    @abstractmethod
    def forward(self, x) -> np.ndarray:
        '''Ejecuta la operación de propagación hacia adelante
        para este nodo..'''

    @abstractmethod
    def backward(self, incoming_grad) -> np.ndarray:
        '''Calcula el gradiente durante la retropropagación
        para este nodo.'''
    
    
class Linear(Node):

    def __init__(self, input_size: int, output_size: int):
        super().__init__(has_weights = True)
        # filas neuronas, columnas pesos
        self.w = np.random.rand(output_size, input_size)
        self.b = np.random.rand(output_size)
        # salida de la capa anterior 
        self.h: Optional[np.ndarray] = None
    
    def forward(self, x) -> np.ndarray:
        self.h = x
        # [:, np.newaxis] tranforma en un vector columna
        self.output = np.dot(self.w, self.h) + self.b[:, np.newaxis]
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

        for i in range(self.output.shape[0]):
            mask = np.ones(self.output.shape[0], dtype=bool)
            mask[i] = False
            _sum = np.sum(incoming_grad[mask] * (- self.output[mask]), axis=0)
            self.grad[i] += _sum * self.output[i]
            
        return self.grad


class CrossEntropy(Node):
    
    def __init__(self,):
        super().__init__()
    
    def forward(self, y_pred, y_real) -> np.ndarray:
        epsilon = 1e-9
        entry_contribution = y_real * np.log(y_pred + epsilon)
        self.output = - np.sum(entry_contribution, axis=0)
        return self.output 

    def backward(self, y_pred, y_real) -> np.ndarray:
        return - (y_real / (y_pred + 1e-9))


class Sequential(Node):
    
    def __init__(self, *layers: Tuple[Node, ...]):
        super().__init__()
        self.layers = layers
        self.params = []
        for layer in layers:
            if layer.has_weights:
                self.params.append(layer)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.T
        if x.ndim == 1: # si solo es un dato se transforma a vector columna
            x = x[:, np.newaxis]
        actual_val = x
        for layer in self.layers:
            actual_val = layer(actual_val)
        return actual_val.T
    
    def backward(self, incoming_grad) -> np.ndarray:
        '''ToDo'''


def make_classification(r0=1,r1=3,k=1000):
    """
    Creacion de los datos
    """
    X1 = [np.array([r0*np.cos(t),r0*np.sin(t)]) for t in range(0,k)]
    X2 = [np.array([r1*np.cos(t),r1*np.sin(t)]) for t in range(0,k)]
    X = np.concatenate((X1,X2))
    n,d = X.shape
    Y = np.zeros(2*k)
    Y[k:] += 1
    noise = np.array([np.random.normal(0,1,2) for i in range(n)])
    X += 0.5*noise
    return X,Y



if __name__ == "__main__":
    x, y = make_classification()
    network = Sequential(Linear(2, 10), ReLU(), Linear(10, 2), Softmax())