import tensorflow as tf
import mnist_inference as mnist
import os
from DiffPrivate_FedLearning import run_differentially_private_federated_averaging
from MNIST_reader import Data

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

    train_op, eval_correct, loss, data_placeholder, labels_placeholder = mnist.mnist_fully_connected_model(Batches, hidden1, hidden2)

    Accuracy_accountant, Delta_accountant, model = \
        run_differentially_private_federated_averaging(loss, train_op, eval_correct, DATA, data_placeholder, labels_placeholder)

'''
def main(_):
    data = Data(FLAGS.save_dir, FLAGS.n)
    train_op, eval_correct, loss = mnist_inference.mnist_fully_connected_model()
    run_differentially_private_federated_averaging(loss, train_op, eval_correct, data)


class Flag:
    def __init__(self, n, b, e, record_privacy, m, sigma, eps, save_dir, log_dir, max_comm_rounds, gm, PrivAgent):
        if not save_dir:
            save_dir = os.getcwd()
        if not log_dir:
            log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/fully_connected_feed')
        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)
        self.n = n
        self.sigma = sigma
        self.eps = eps
        self.m = m
        self.b = b
        self.e = e
        self.record_privacy = record_privacy
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.max_comm_rounds = max_comm_rounds
        self.gm = gm
        self.PrivAgentName = PrivAgent.Name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--PrivAgentName',
        type=str,
        default='default_Priv_Agent',
        help='Sets the name of the used Privacy agent'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=100,
        help='Total Number of clients participating'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0,
        help='The gm variance parameter; will not affect if Priv_agent is set to True'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=8,
        help='Epsilon'
    )
    parser.add_argument(
        '--m',
        type=int,
        default=0,
        help='Number of clients participating in a round'
    )
    parser.add_argument(
        '--b',
        type=float,
        default=10,
        help='Batches per client'
    )
    parser.add_argument(
        '--e',
        type=int,
        default=4,
        help='Epochs per client'
    )
    parser.add_argument(
        '--record_privacy',
        type=bool,
        default=True,
        help='Epochs per client'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=os.getcwd(),
        help='Directory.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--max_comm_rounds',
        type=int,
        default=3000,
        help='Maximum number of communication rounds'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
'''