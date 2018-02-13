from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import sys
import os.path
from six.moves import xrange
import tensorflow as tf
from Helper_Functions import Vname_to_FeedPname, Vname_to_Pname, check_validaity_of_FLAGS, create_save_dir, \
    global_step_creator, Data, load_from_directory_or_initialize, bring_Accountant_up_to_date, save_progress, \
    WeightsAccountant, print_loss_and_accuracy, print_new_comm_round, PrivAgent
import math
import os

FLAGS = None


def run_differentially_private_federated_averaging(loss, train_op, eval_correct, DATA, PrivacyAgent = None,
                                                   N=100, B=10, E=4, Record_privacy=True, m=0, Sigma=0, eps=8,
                                                   save_dir=None, log_dir=None,
                                                   Max_comm_rounds=3000, GM = True
                                                   ):

    '''

    :param loss: TENSORFLOW node that computes the current loss
    :param train_op: TENSORFLOW Training_op
    :param eval_correct: TENSORFLOW node that evaluates teh number of correct predictions
    :param PrivacyAgent: A class instance that has callabels .get_m(r) .get_Sigma(r) .get_bound(), where r is the communication round.
    :param DATA: A class instance with attributes .data_set, .client_set and .validation_set. data_set is a list of data points, client_set is a list of indices, mapping data points to clients.
    :param PrivAgentName: The name of the specified privacyAgent, this is for saving purposes.
    :param N: Number of participating clients
    :param B: Batchsize
    :param E: Epochs to run on each client
    :param Record_privacy: Whether to record the privacy or not
    :param m: If specified, a privacyAgent is not used, instead the parameter is kept constant
    :param Sigma: If specified, a privacyAgent is not used, instead the parameter is kept constant
    :param eps: The epsilon for epsilon-delta privacy
    :param save_dir: Directory to store the process
    :param log_dir: directory to store the graph
    :param Max_comm_rounds:
    :param GM:
    :return:
    '''
    if PrivacyAgent == None:
        PrivacyAgent = PrivAgent
    FLAGS = flag(N, B, E, Record_privacy, m, Sigma, eps, save_dir, log_dir, Max_comm_rounds, GM)
    FLAGS = check_validaity_of_FLAGS(FLAGS)
    FLAGS.PrivAgentName = PrivAgent.Name

    # At this point, FLAGS.save_dir specifies both; where we save and where we assume the data is stored
    save_dir = create_save_dir(FLAGS)

    increase_global_step, set_global_step = global_step_creator()

    # - model_placeholder : a dictionary in which there is a placeholder stored for every trainable variable defined
    #                       in the tensorflow graph. Each placeholder corresponds to one trainable variable and has
    #                       the same shape and dtype as that variable. in addition, the placeholder has the same
    #                       name as the Variable, but a '_placeholder:0' added to it. The keys of the dictionary
    #                       correspond to the name of the respective placeholder

    model_placeholder = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                                 [tf.placeholder(name=Vname_to_Pname(var),
                                                 shape=var.shape,
                                                 dtype=tf.float32)
                                  for var in tf.trainable_variables()]))

    # - assignments : Is a list of nodes. when run, all trainable variables are set to the value specified through
    #                 the placeholders in 'model_placeholder'.

    assignments = [tf.assign(var, model_placeholder[Vname_to_FeedPname(var)]) for var in
                   tf.trainable_variables()]

    # load_from_directory_or_initialize checks whether there is a model at 'save_dir' corresponding to the one we
    # are building. If so, training is resumed, if not, it returns:  - model = []
    #                                                                - Accuracy_accountant = []
    #                                                                - Delta_accountant = []
    #                                                                - real_round = 0
    # And initializes a Differential_Privacy_Accountant as Acc
    model, Accuracy_accountant, Delta_accountant, Acc, real_round, FLAGS, Computed_deltas = \
        load_from_directory_or_initialize(save_dir, FLAGS)

    # - m : amount of clients participating in a round
    # - Sigma : variable for the Gaussian Mechanism.
    # Both will only be used if no Privacy_Agent is deployed.

    m = FLAGS.m
    Sigma = FLAGS.Sigma

    ################################################################################################################
    ################################################################################################################

    # Usual Tensorflow...

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    sess.run(init)

    ################################################################################################################
    ################################################################################################################

    # If there was no loadable model, we initialize a model:
    # - model : dictionary having as keys the names of the placeholders associated to each variable. It will serve
    #           as a feed_dict to assign values to the placeholders which are used to set the variables to
    #           specific values.

    if not model:
        model = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                         [sess.run(var) for var in tf.trainable_variables()]))
        model['global_step_placeholder:0'] = 0

        real_round = 0

        Weights_Accountant = []

    # If a model is loaded, we have to get the privacy accountant up to date. This means, that we have to
    # iterate the privacy accountant over all the m, sigmas that correspond to already completed communication
    # rounds. The Problem is, we do not kno

    if real_round > 0:
        if FLAGS.relearn == False:
            bring_Accountant_up_to_date(Acc, sess, real_round, PrivacyAgent, FLAGS)

    ################################################################################################################
    ################################################################################################################

    for r in xrange(FLAGS.Max_comm_rounds):

        # First, we check whether we are loading a model, if so, we have to skip the first allocation, as it took place
        # already.
        if not (FLAGS.loaded == True and r == 0):
            # Setting the trainable Variables in the graph to the values stored in feed_dict 'model'
            sess.run(assignments, feed_dict=model)

            # create a feeddict holding the validation set.

            feed_dict = {'images_placeholder:0': DATA.validation_set[0],
                         'labels_placeholder:0': DATA.validation_set[1]}

            # compute the loss on the validation set.
            global_loss = sess.run(loss, feed_dict=feed_dict)
            count = sess.run(eval_correct, feed_dict=feed_dict)
            accuracy = float(count) / float(len(DATA.validation_set[0]))
            Accuracy_accountant.append(accuracy)

            print_loss_and_accuracy(global_loss, accuracy)

        if Delta_accountant[-1] > PrivacyAgent.get_bound() or math.isnan(Delta_accountant[-1]):
            print('************** The last step exhausted the privacy budget **************')
            if not math.isnan(Delta_accountant[-1]):
                try:
                    None
                finally:
                    save_progress(save_dir, model, Delta_accountant + [float('nan')],
                                  Accuracy_accountant + [float('nan')], PrivacyAgent, FLAGS)
                return Accuracy_accountant, Delta_accountant, model
        else:
            try:
                None
            finally:
                save_progress(save_dir, model, Delta_accountant, Accuracy_accountant, PrivacyAgent, FLAGS)

        ############################################################################################################
        ##################################### Start communication round ############################################

        real_round = real_round + 1

        print_new_comm_round(real_round)

        if FLAGS.PrivAgent:
            m = int(PrivacyAgent.get_m(int(real_round)))
            Sigma = PrivacyAgent.get_Sigma(int(real_round))

        print('Clients participating: ' + str(m))

        # Randomly choose a total of m (out of N) client-indices that participate in this round
        perm = np.random.permutation(FLAGS.N)
        S = perm[0:m].tolist()
        participating_clients = [DATA.client_set[k] for k in S]

        # For each client c (out of the m chosen ones):
        for c in range(m):

            # Assign the global model and set the global step. This is obsolete when the first client trains,
            # but as soon as the next clients trains, all progress allocated before, has to be discarded and the
            # trainable variables reset to the values specified in 'model'
            sess.run(assignments + [set_global_step], feed_dict=model)

            # allocate a list, holding the training data associated to client c and split into batches.
            data_ind = np.split(participating_clients[c], FLAGS.B, 0)

            # e = Epoch
            for e in xrange(int(FLAGS.E)):
                for step in xrange(len(data_ind)):
                    real_step = sess.run(increase_global_step)
                    batch_ind = data_ind[step]

                    # Fill a feed dictionary with the actual set of images and labels
                    # for this particular training step.
                    feed_dict = {'images_placeholder:0': DATA.data_set[[int(j) for j in batch_ind]][:, 1:],
                                 'labels_placeholder:0': DATA.data_set[[int(j) for j in batch_ind]][:, 0]}

                    # Run one optimization step.
                    _ = sess.run([train_op], feed_dict=feed_dict)

            if c == 0:

                # If we just trained the first client in a comm_round, We override the old Weights_Accountant (or,
                # if this was the first comm_round, we allocate a new one. The Weights_accountant keeps track of
                # all client updates throughout a communication round.
                Weights_Accountant = WeightsAccountant(sess, model, Sigma, real_round)
            else:
                # Allocate the client update
                Weights_Accountant.allocate(sess)

        #################################### End communication round ###############################################
        ############################################################################################################

        print('......Communication round %s completed' % str(real_round))
        # Compute a new model according to the updates and the Gaussian mechanism specifications from FLAGS
        # Also compute delta; the probability of Epsilon-Differential Privacy being broken by allocating the model.
        model, delta = Weights_Accountant.Update_via_GaussianMechanism(sess, Acc, FLAGS, Computed_deltas)

        # Save delta.
        Delta_accountant.append(delta)

        # Set the global_step to the current step of the last client, such that the next clients can feed it into
        # the learning rate.
        model['global_step_placeholder:0'] = real_step

        # PRINT the progress and stage of affairs.
        print(' - Epsilon-Delta Privacy:' + str([FLAGS.eps, delta]))

    return [], [], []


def main(_):
    data = Data(FLAGS.save_dir, FLAGS.N)
    run_differentially_private_federated_averaging(PrivacyAgent=PrivAgent(FLAGS.N), FLAGS=FLAGS, DATA=data)


class flag:
    def __init__(self, N, B, E, Record_privacy, m, Sigma, eps, save_dir, log_dir, Max_comm_rounds, GM):
        if not save_dir:
            save_dir = os.getcwd()
        if not log_dir:
            log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/fully_connected_feed')
        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)
        self.PrivAgentName = PrivAgentName
        self.N = N
        self.Sigma = Sigma
        self.eps = eps
        self.m = m
        self.B = B
        self.E = E
        self.Record_privacy = Record_privacy
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.Max_comm_rounds = Max_comm_rounds
        self.GM = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--PrivAgentName',
        type=str,
        default='default_Priv_Agent',
        help='Sets the name of the used Privacy agent'
    )
    parser.add_argument(
        '--N',
        type=int,
        default=100,
        help='Total Number of clients participating'
    )
    parser.add_argument(
        '--Sigma',
        type=float,
        default=0,
        help='The GM variance parameter; will not affect if Priv_agent is set to True'
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
        '--B',
        type=float,
        default=10,
        help='Batches per client'
    )
    parser.add_argument(
        '--E',
        type=int,
        default=4,
        help='Epochs per client'
    )
    parser.add_argument(
        '--Record_privacy',
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
        '--Max_comm_rounds',
        type=int,
        default=3000,
        help='Maximum number of communication rounds'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
