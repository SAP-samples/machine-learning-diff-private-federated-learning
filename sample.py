import mnist_inference as mnist
from Redone import run_differentially_private_federated_averaging
import os
from Helper_Functions import Data, PrivAgent

N = 100
hidden1 = 600
hidden2 = 100
Batches = 10
save_dir = os.getcwd()
DATA = Data(save_dir, N)
train_op, eval_correct, loss = mnist.mnist_cnn_model(Batches)

Accuracy_accountant, Delta_accountant, model = \
    run_differentially_private_federated_averaging(loss, train_op, eval_correct,
                                                   PrivacyAgent[i], DATA, save_dir=save_dir+'/CNN', B=60)