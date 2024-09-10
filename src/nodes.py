'''Modulo para la implementación de nodos en una red neuronal.'''

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np

class Node(ABC):
    '''Clase base abstracta para nodos en una red neuronal.'''

    def __init__(self, *previous_nodes):
        self.previous_nodes = previous_nodes
        self.output: Union[np.ndarray, float] = None # type: ignore
        self.grad: Union[np.ndarray, Tuple[np.ndarray, ...], None] = None

    def __call__(self, x) -> Union[np.ndarray, float]:
        return self.forward(x)

    @abstractmethod
    def forward(self, x) -> Union[np.ndarray, float]:
        '''Ejecuta la operación de propagación hacia adelante
        para este nodo..'''

    @abstractmethod
    def backward(self, incoming_grad) -> None:
        '''Calcula el gradiente durante la retropropagación
        para este nodo.'''


class PreActivation(Node):
    '''Nodo que calcula el valor de pre-activación
    (suma ponderada más sesgo)'''

    def __init__(self, dimension: int, *previous_nodes):
        super().__init__(*previous_nodes)
        self.w = np.random.uniform(0, 1, size=dimension)
        self.b = np.random.random()
        self.x: Optional[np.ndarray] = None

    def forward(self, x):
        self.x = x
        self.output = np.dot(self.w, x.T) + self.b
        return self.output

    def backward(self, incoming_grad):

        # multiplicacion de entrada for fila en
        # caso de que se aplique a mas de un dato
        grad_w = self.x * incoming_grad[:, np.newaxis]
        grad_b = incoming_grad
        self.grad = grad_w, grad_b


class Sigmoid(Node):
    '''Nodo que aplica la función de activación sigmoide
    a la salida de un nodo de operación anterior.'''

    def __init__(self, previous: Node):
        super().__init__(previous)

    def forward(self, x):
        prev_node = self.previous_nodes[0]
        self.output = 1 / (1 + np.exp(- prev_node(x)))
        return self.output

    def backward(self, incoming_grad):

        # parcial con respecto a la preactivation
        # sigm'(x) = sigm(x) * (1 - sigm(x))
        self.grad = incoming_grad * (self.output * (1 - self.output))
        self.previous_nodes[0].backward(self.grad)


class BinCrossEntropy(Node):
    '''Nodo que calcula la entropía cruzada binaria entre las
    predicciones del modelo y las etiquetas verdaderas.'''

    def __init__(self, previous: Node):
        super().__init__(previous)

    def forward(self, y): # type: ignore pylint: disable=arguments-renamed
        f_x = self.previous_nodes[0].output
        self.output = -(y * np.log(f_x) + (1 - y) * np.log(1 - f_x))
        return self.output

    def backward(self, y): # type: ignore pylint: disable=arguments-renamed
        prev_output = self.previous_nodes[0].output

        # parcial con respecto a la activacion
        # cuando y == 1 entonces -1/x pero 1/(1-x) en otro caso
        self.grad = np.where(y == 1,
                             - 1 / prev_output,
                             1 / (1 - prev_output))
        self.previous_nodes[0].backward(self.grad)


if __name__ == "__main__":
    preact = PreActivation(2)
    logistic_reg = Sigmoid(preact)

    data_set = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    data_set_labels = np.array([0, 0, 0, 1])
    print("dataset:\n", data_set)
    print("etiquetas:", data_set_labels)

    # regresion logistica para cada entrada
    # del dataset
    print("regresion del dataset: ", logistic_reg(data_set))

    loss = BinCrossEntropy(logistic_reg)
    # error de cada entrada del dataset
    print("error: ", loss(data_set_labels))


    loss.backward(data_set_labels)
    print("grad_w:\n", preact.grad[0])  # type: ignore
    print("grad_b:", preact.grad[1])    # type: ignore
