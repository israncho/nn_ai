from abc import ABC, abstractmethod
from typing import Union
import numpy as np

class Node(ABC):

    def __init__(self, *previous_nodes,
                 output: Union[np.ndarray, float] = 0):

        self.previous_nodes = previous_nodes
        self.output: Union[np.ndarray, float] = output

    def __call__(self, *args) -> Union[np.ndarray, float]:
        return self.forward(*args)

    @abstractmethod
    def forward(self, *args) -> Union[np.ndarray, float]:
        '''Ejecuta la operación con la entrada proporcionada.'''

    @abstractmethod
    def backward(self, *args) -> None:
        ''' To Do '''


class Weights(Node):
    '''Nodo que contiene pesos inicializados aleatoriamente.'''

    def __init__(self, dimension: int):
        w = np.random.uniform(0, 1, size=dimension)
        super().__init__(output=w)

    def forward(self, *args):
        return self.output

    def backward(self, *args):
        pass


class Bias(Node):
    '''Nodo que contiene un valor de sesgo inicializado aleatoriamente.'''

    def __init__(self):
        super().__init__(output=np.random.random())

    def forward(self, *args):
        return self.output

    def backward(self, *args):
        pass


class PreActivation(Node):
    '''Nodo que calcula el valor de pre-activación
    (suma ponderada más sesgo)'''

    def __init__(self, w: Weights, b: Bias):
        super().__init__(w, b)
        self.x: Union[np.ndarray, float] = 0

    def forward(self, *args):
        w, b = self.previous_nodes
        self.x = x = args[0]
        self.output = np.dot(w(), x.T) + b()
        return self.output

    def backward(self, *args):
        w, b = self.previous_nodes
        child_partial = args[0]

        # multiplicacion de entrada for fila en
        # caso de que se aplique a un lote
        grad_w = self.x * child_partial[:, np.newaxis]
        grad_b = child_partial
        w.backward(grad_w)
        b.backward(grad_b)


class Sigmoid(Node):
    '''Nodo que aplica la función de activación sigmoide
    a la salida de un nodo de operación anterior.'''

    def __init__(self, previous: Node):
        super().__init__(previous)

    def forward(self, *args):
        prev_node = self.previous_nodes[0]
        x = args[0]
        self.output = 1 / (1 + np.exp(- prev_node(x)))
        return self.output

    def backward(self, *args):
        child_partial = args[0]
        prev_output = self.previous_nodes[0].output
        e_x = np.exp(-prev_output)
        partial_preactivation = child_partial * (e_x / (1 + e_x)**2)
        self.previous_nodes[0].backward(partial_preactivation)


class BinCrossEntropy(Node):
    '''Nodo que calcula la entropía cruzada binaria entre las
    predicciones del modelo y las etiquetas verdaderas.'''

    def __init__(self, previous: Node):
        super().__init__(previous)

    def forward(self, *args):
        x, y = args
        prev_node = self.previous_nodes[0]
        f_x = prev_node(x)
        self.output = -(y * np.log(f_x) + (1 - y) * np.log(1 - f_x))
        return self.output

    def backward(self, *args):
        y = args[0]
        prev_output = self.previous_nodes[0].output
        partial_activation = (prev_output - y) / (prev_output - prev_output**2)
        self.previous_nodes[0].backward(partial_activation)


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
