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


    print('Stochastic gradient descent:')
    preact = PreActivation(2)
    sigm = Sigmoid(preact)
    loss = BinCrossEntropy(sigm)
    loss_history = []
    lr = 0.1

    for _ in range(100):
        sigm(preact(x_eval))
        eval_loss = np.mean(loss.forward(y_eval))
        sigm(preact(x_train))
        train_loss = np.mean(loss.forward(y_train))
        loss_history.append((train_loss, eval_loss))
        for x_i, y_i in zip(x_train, y_train):
            sigm(preact(x_i)) # forward
            loss.backward(y_i) # backward
            grad_w, grad_b = preact.grad
            preact.w -= lr * grad_w
            preact.b -= lr * grad_b

    for i, (t_loss, ev_loss) in enumerate(loss_history[::10]):
        print("epoch:", i*10,"\ttrain_loss:", t_loss, 'ev_loss:', ev_loss)

    print("\n")
    y_prob = sigm(preact(x_eval))
    y_pred = (y_prob > 0.5).astype(int)
    print(classification_report(y_eval, y_pred))

    #-----------------------------------------------------------------

    print('\n\nBatch gradient descent:')

    preact = PreActivation(2)
    sigm = Sigmoid(preact)
    loss = BinCrossEntropy(sigm)
    loss_history = []
    lr = 0.1
    for _ in range(100):
        sigm(preact(x_eval))
        eval_loss = np.mean(loss.forward(y_eval))
        sigm(preact(x_train))
        train_loss = np.mean(loss.forward(y_train))
        loss_history.append((train_loss, eval_loss))

        sigm(preact(x_train))
        loss.backward(y_train)

        grads_w, grads_b = preact.grad

        grads_w = np.sum(grads_w, axis=0)

        grads_b = np.sum(grads_b, axis=0)

        preact.w -= lr * grad_w
        preact.b -= lr * grad_b

    for i, (t_loss, ev_loss) in enumerate(loss_history[::10]):
        print("epoch:", i*10,"\ttrain_loss:", t_loss, 'ev_loss:', ev_loss)

    print("\n")
    y_prob = sigm(preact(x_eval))
    y_pred = (y_prob > 0.5).astype(int)
    print(classification_report(y_eval, y_pred))
