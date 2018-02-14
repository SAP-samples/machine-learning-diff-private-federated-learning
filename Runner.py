import mnist_inference as mnist
from Redone import run_differentially_private_federated_averaging, flag
import numpy as np
from Helper_Functions import Data, PrivAgent
import os
import tensorflow as tf


N = 100
hidden1 = 600
hidden2 = 100
Batches = 10
save_dir = os.getcwd()


PrivacyAgent = []

for j in range(10):
    PrivacyAgent.append(PrivAgent(100, '_'+str(j)))

PrivacyAgent[0].m = [20]*100
PrivacyAgent[1].m = [25]*100
PrivacyAgent[2].m = [30]*100
PrivacyAgent[3].m = [35]*100
PrivacyAgent[4].m = [40]*100
PrivacyAgent[5].m = [45]*100
PrivacyAgent[6].m = [20,30,40,45,45,45,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
PrivacyAgent[7].m = [20,30,40,40,45,45,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
PrivacyAgent[8].m = [10,30,40,45,45,45,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
PrivacyAgent[9].m = [20,20,30,45,45,45,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]

Acc = []
Del = []

DATA = Data(save_dir, N)

for i in range(10):
    Acc_temp = []
    Del_temp = []
    for j in range(10):

        PrivAgentName = 'Priv_Agent'+str(i)
        with tf.Graph().as_default():

            # A train operation, an evaluating operation, allocating a loss.
            train_op, eval_correct, loss, data_placeholder, labels_placeholder = mnist.mnist_cnn_model(Batches)

            Accuracy_accountant, Delta_accountant, model = \
                run_differentially_private_federated_averaging(loss, train_op, eval_correct, data_placeholder,
                                                               labels_placeholder,
                                                               PrivacyAgent[i], DATA, save_dir=save_dir+'/CNN', B=60)
            Acc_temp.append(Accuracy_accountant)
            Del_temp.append(Delta_accountant)


Acc_temp = np.asarray(Acc_temp)
Acc_temp = np.mean(Acc_temp,0)
Acc.append(Acc_temp)
Del.append(Del_temp)

print(Acc)
print(Del)