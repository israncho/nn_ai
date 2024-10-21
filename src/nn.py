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


class PreActivation(Node):
    '''Nodo que calcula el valor de pre-activación
    (suma ponderada más sesgo)'''

    def __init__(self, input_size: int):
        super().__init__(has_weights=True)
        self.w = np.random.uniform(0, 1, size=input_size)
        self.b = np.random.random()
        # salida de la capa anterior
        self.h: Optional[np.ndarray] = None

    def forward(self, x):
        self.h = x
        self.output = np.dot(self.w, x.T) + self.b
        return self.output

    def backward(self, incoming_grad):

        # multiplicacion de entrada for fila en
        # caso de que se aplique a mas de un dato
        if not np.isscalar(incoming_grad):
            incoming_grad = incoming_grad[:, np.newaxis]

        grad_w = self.h * incoming_grad
        grad_b = incoming_grad
        self.grad = grad_w, grad_b
        return self.w


class ReLU(Node):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.output = max(0, x)
        return self.output
    
    def backward(self, incoming_grad):
        pass