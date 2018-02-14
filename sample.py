import tensorflow as tf
import mnist_inference as mnist
import os
from DiffPrivate_FedLearning import run_differentially_private_federated_averaging
from Helper_Functions import Data

# Specs for the model that we would like to train in differentially private federated fashion:
hidden1 = 600
hidden2 = 100

# Specs for the differentially private federated fashion learning process.
N = 100
Batches = 10
save_dir = os.getcwd()

# A data object that already satisfies client structure and has the following attributes:
# DATA.data_set : A list of labeld training examples.
# DATA.client_set : A
DATA = Data(save_dir, N)

with tf.Graph().as_default():

    # Building the model that we would like to train in differentially private federated fashion.
    # We will need the tensorflow training operation for that model, its loss and an evaluation method:

    train_op, eval_correct, loss = mnist.mnist_fully_connected_model(Batches, hidden1, hidden2)

    Accuracy_accountant, Delta_accountant, model = \
        run_differentially_private_federated_averaging(loss, train_op, eval_correct, DATA)
