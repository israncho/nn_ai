import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nodes import PreActivation, Sigmoid, BinCrossEntropy

if __name__ == "__main__":
    x, y = make_classification(n_samples=1000,
                               n_features=2,
                               n_redundant=0,
                               n_informative=2,
                               random_state=10)
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.3)


    preact = PreActivation(2)
    logistic_reg = Sigmoid(preact)
    loss = BinCrossEntropy(logistic_reg)
    loss_history = []
    lr = 0.1
    for _ in range(100):
        logistic_reg(x)
        loss_history.append(np.mean(loss(y)))
        for x_i, y_i in zip(x_train, y_train):
            logistic_reg(x_i) # forward
            loss.backward(y_i) # backward
            grad_w, grad_b = preact.grad
            preact.w -= lr * grad_w
            preact.b -= lr * grad_b

    for i, loss in enumerate(loss_history[:10]):
        print("epoca:",i ,"\terror:", loss)

    print("\n")
    y_prob = logistic_reg(x_eval)
    y_pred = (y_prob > 0.5).astype(int)
    print(classification_report(y_eval, y_pred))
