from nodes import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def LogisticRegression(x,y, learning_rate, epochs):
    """ Función que realiza una regresión logística 
        Recibe datos de entrada x y sus etiquetas y
        Recibe una tasa de aprendizaje y el número de épocas 
        a entrenar.
        Regresa un nodo tipo Sigmoide
    """
    # Nodo de preactivación lineal
    preactivation = PreActivation(dimension)
    # Nodo de activación sigmoid
    activation = Sigmoid(preactivation)
    # Función objetivo o de pérdida
    loss_function = BinCrossEntropy(activation)

    for i in range(epochs):
        # Genera la sigmoide de la combinación lineal de los datos
        activation.forward(x_train)
        # Calcula la derivada de la función objetivo
        loss_function.backward(y_train)

        # Guarda las derivadas
        grad_w, grad_b = preactivation.grad

        # Descenso por gradiente
        for grad_wi, grad_bi in zip(grad_w,grad_b):
            preactivation.w -= learning_rate * grad_wi
            preactivation.b -= learning_rate * grad_bi
    
    print(preactivation)
    return activation


if __name__ == "__main__":
    x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=10)
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.3)
    dimension = 2

    learning_rate = 0.03
    epochs = 100

    solution_node = LogisticRegression(x_train, y_train, learning_rate, epochs)
    report = classification_report(y_eval, solution_node.classify(x_eval))
    print(report)
