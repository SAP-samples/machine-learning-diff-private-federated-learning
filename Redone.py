from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import csv
import argparse
import os
import sys
import os.path
import mnist_inference as mnist
from six.moves import xrange
import tensorflow as tf
import pickle
from Helper_Functions import placeholder_inputs, Vname_to_FeedPname, Vname_to_Pname
from accountant import GaussianMomentsAccountant
import math
import os


# Basic model parameters as external flags.
FLAGS = None
IMAGE_PIXELS = 28 * 28
NUM_CLASSES = 10

class WeightsAccountant:
    def __init__(self,sess,model,Sigma, real_round):

        self.Weights = [np.expand_dims(sess.run(v), -1) for v in tf.trainable_variables()]
        self.keys = [Vname_to_FeedPname(v) for v in tf.trainable_variables()]

        # The trainable parameters are [q x p] matrices, we expand them to [q x p x 1] in order to later stack them
        # along the third dimension.

        # Create a list out of the model dictionary in the order in which the graph holds them:

        self.global_model = [model[k] for k in self.keys]
        self.Sigma = Sigma
        self.Updates = []
        self.median = []
        self.Norms = []
        self.ClippedUpdates = []
        self.m = 0.0
        self.num_weights = len(self.Weights)
        self.round = real_round

    def allocate(self, sess):

        self.Weights = [np.concatenate((self.Weights[i], np.expand_dims(sess.run(tf.trainable_variables()[i]), -1)), -1)
                        for i in range(self.num_weights)]

        # The trainable parameters are [q x p] matrices, we expand them to [q x p x 1] in order to stack them
        # along the third dimension to the already allocated older variables. We therefore have a list of 6 numpy arrays
        # , each numpy array having three dimensions. The last dimension is the one, the individual weight
        # matrices are stacked along.

    def compute_updates(self):

        # To compute the updates, we subtract the global model from each individual weight matrix. Note:
        # self.Weights[i] is of size [q x p x m], where m is the number of clients whose matrices are stored.
        # global_model['i'] is of size [q x p], in order to broadcast correctly, we have to add a dim.

        self.Updates = [self.Weights[i]-np.expand_dims(self.global_model[i], -1) for i in range(self.num_weights)]
        self.Weights = None

    def compute_norms(self):

        # The norms List shall have 6 entries, each of size [1x1xm], we keep the first two dimensions because
        # we will later broadcast the Norms onto the Updates of size [q x p x m]

        self.Norms = [np.sqrt(np.sum(
            np.square(self.Updates[i]), axis=tuple(range(self.Updates[i].ndim)[:-1]),keepdims=True)) for i in range(self.num_weights)]

    def clip_updates(self):
        self.compute_updates()
        self.compute_norms()

        # The median is a list of 6 entries, each of size [1x1x1],

        self.median = [np.median(self.Norms[i], axis=-1, keepdims=True) for i in range(self.num_weights)]

        # The factor is a list of 6 entries, each of size [1x1xm]

        factor = [self.Norms[i]/self.median[i] for i in range(self.num_weights)]
        for i in range(self.num_weights):
            factor[i][factor[i] > 1.0] = 1.0

        self.ClippedUpdates = [self.Updates[i]/factor[i] for i in range(self.num_weights)]

    def Update_via_GaussianMechanism(self, sess, Acc, FLAGS, Computed_deltas):
        self.clip_updates()
        self.m = float(self.ClippedUpdates[0].shape[-1])
        MeanClippedUpdates = [np.mean(self.ClippedUpdates[i], -1) for i in range(self.num_weights)]

        GaussianNoise = [(1.0/self.m * np.random.normal(loc=0.0, scale=float(self.Sigma * self.median[i]), size=MeanClippedUpdates[i].shape)) for i in range(self.num_weights)]

        Sanitized_Updates = [MeanClippedUpdates[i]+GaussianNoise[i] for i in range(self.num_weights)]

        New_weights = [self.global_model[i]+Sanitized_Updates[i] for i in range(self.num_weights)]

        New_model = dict(zip(self.keys, New_weights))

        t = Acc.accumulate_privacy_spending(0, self.Sigma, self.m)
        delta = 1
        if FLAGS.Record_privacy == True:
            if FLAGS.relearn == False:
                # I.e. we never learned a complete model before and have therefore never computed all deltas.
                for j in range(len(self.keys)):
                    sess.run(t)
                r = Acc.get_privacy_spent(sess, [FLAGS.eps])
                delta = r[0][1]
                # I.e. we have computed a complete model before and can reuse the deltas from that time.
            else: delta = Computed_deltas[self.round]
        return New_model, delta


class PrivAgent:
    def __init__(self, N):
        self.N = N
        if N == 100:
            self.m = [20,30,40,45,45,45,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
            self.Sigma = [1]*24
            self.bound = 0.001
        if N == 1000:
            self.m = []
            self.Sigma = []
            self.bound = 0.00001
        if N == 10000:
            self.m = []
            self.Sigma = []
            self.bound = 0.000001
        if(N != 100 and N != 1000 and
          N != 10000 ):
            print('!!!!!!! YOU CAN ONLY USE THE PRIVACY AGENT FOR N = 100, 1000 or 10000 !!!!!!!')

    def get_m (self, r):
        return self.m[r]

    def get_Sigma (self, r):
        return self.Sigma[r]

    def get_bound(self):
        return self.bound



def create_save_dir(FLAGS):
    '''
    :return: Returns a path that is used to store training progress; the path also identifies the chosen setup uniquely.
    '''
    raw_directory = FLAGS.save_dir + '/'
    if FLAGS.GM: gm_str = 'Dp/'
    else: gm_str = 'non_Dp/'
    if FLAGS.PrivAgent:
        model = gm_str + 'N_' + str(FLAGS.N) + '/Epochs_' + str(
            int(FLAGS.E)) + '_Batches_' + str(int(FLAGS.B)) + '/hidden1_' + str(FLAGS.hidden1) + '__hidden2_' + str(
            FLAGS.hidden2)
        return raw_directory + str(model) + '/' + FLAGS.PrivAgentName
    else:
        model = gm_str + 'N_' + str(FLAGS.N) + '/Sigma_' + str(FLAGS.Sigma) + '_C_'+str(FLAGS.m)+'/Epochs_' + str(
            int(FLAGS.E)) + '_Batches_' + str(int(FLAGS.B)) + '/hidden1_' + str(FLAGS.hidden1) + '__hidden2_' + str(
            FLAGS.hidden2)
        return raw_directory + str(model)

def get_data(FLAGS):
    '''
    
    :return: returns the directory where the data_set is stored 
    '''
    raw_directory = FLAGS.save_dir + '/DATA/'
    data_set = pickle.load(open(raw_directory + 'Sorted_MNIST.pkl', 'rb'))
    client_set = pickle.load(open(raw_directory + 'clients/' + str(FLAGS.N) + '_clients.pkl', 'rb'))
    test_set = pickle.load(open(raw_directory + 'test_set.pkl', 'rb'))

    return data_set, client_set, test_set


def load_from_directory_or_initialize(directory, FLAGS):

    '''
    This function look for a model that corresponds to the characteristics specified and loads potential progress.
    If it does not find any model or progress, it initializes a new model.
    :param directory: STRING: the directory where to look for models and progress.
    :param FLAGS: CLASS INSTANCE: holds general trianing params
    :param PrivacyAgent:
    :return:
    '''

    Accuracy_accountant = []
    Delta_accountant = [0]
    model = []
    real_round = 0
    Acc = GaussianMomentsAccountant(FLAGS.N)
    FLAGS.loaded = False
    FLAGS.relearn = False
    Computed_Deltas = []

    if not os.path.isfile(directory + '/model.pkl'):
        # If there is no model stored at the specified directory, we initialize a new one!
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('No loadable model found. All updates stored at: ' + directory)
        print('... Initializing a new model ...')


    else:
        # If there is a model, we have to check whether:
        #  - We learned a model for the first time, and interrupted; in that case: resume learning:
        #               set FLAGS.loaded = TRUE
        #  - We completed learning a model and want to learn a new one with the same parameters, i.o. to average accuracies:
        #       In this case we would want to initialize a new model; but would like to reuse the delta's already
        #       computed. So we will load the deltas.
        #               set FLAGS.relearn = TRUE
        #  - We completed learning models and want to resume learning model; this happens if the above process is
        # interrupted. In this case we want to load the model; and reuse the deltas.
        #               set FLAGS.loaded = TRUE
        #               set FLAGS.relearn = TRUE
        if os.path.isfile(directory + '/specs.csv'):
            with open(directory + '/specs.csv', 'rb') as csvfile:
                reader = csv.reader(csvfile)
                Lines = []
                for line in reader:
                    Lines.append([float(j) for j in line])

                Accuracy_accountant = Lines[-1]
                Delta_accountant = Lines[1]

            if math.isnan(Delta_accountant[-1]):
                Computed_Deltas = Delta_accountant
                # This would mean that learning was finished at least once, i.e. we are relearning.
                # We shall thus not recompute the deltas, but rather reuse them.
                FLAGS.relearn = True
                if math.isnan(Accuracy_accountant[-1]):
                    # This would mean that we finished learning the latest model.
                    print('A model identical to that specified was already learned. Another one is learned and appended')
                    Accuracy_accountant = []
                    Delta_accountant = [0]
                else:
                    # This would mean we already completed learning a model once, but the last one stored was not completed
                    print('A model identical to that specified was already learned. For a second one learning is resumed')
                    # We set the delta accountant accordingly
                    Delta_accountant = Delta_accountant[:len(Accuracy_accountant)]
                    # We specify that a model was loaded
                    real_round = len(Accuracy_accountant) - 1
                    fil = open(directory + '/model.pkl', 'rb')
                    model = pickle.load(fil)
                    fil.close()
                    FLAGS.loaded = True
                return model, Accuracy_accountant, Delta_accountant, Acc, real_round, FLAGS, Computed_Deltas
            else:
                # This would mean that learning was never finished, i.e. the first time a model with this specs was
                # learned got interrupted.
                real_round = len(Accuracy_accountant) - 1
                fil = open(directory + '/model.pkl', 'rb')
                model = pickle.load(fil)
                fil.close()
                FLAGS.loaded = True
        else:
            print('there seems to be a model, but no saved progress. Fix that.')
            raise KeyboardInterrupt
    return model, Accuracy_accountant, Delta_accountant, Acc, real_round, FLAGS, Computed_Deltas


def save_progress(save_dir, model, Delta_accountant, Accuracy_accountant, PrivacyAgent, FLAGS):
    '''
    This function saves our progress either in an existing file structure or writes a new file.
    :param save_dir: STRING: The directory where to save the progress.
    :param model: DICTIONARY: The model that we wish to save.
    :param Delta_accountant: LIST: The list of deltas that we allocared so far.
    :param Accuracy_accountant: LIST: The list of accuracies that we allocated so far.
    :param PrivacyAgent: CLASS INSTANCE: The privacy agent that we used (specifically the m's that we used for Federated training.)
    :param FLAGS: CLASS INSTANCE: The FLAGS passed to the learning procedure.
    :return: nothing
    '''
    filehandler = open(save_dir + '/model.pkl', "wb")
    pickle.dump(model, filehandler)
    filehandler.close()

    if FLAGS.relearn == False:
        # I.e. we know that there was no progress stored at 'save_dir' and we create a new csv-file that
        # Will hold the accuracy, the deltas, the m's and we also save the model learned as a .pkl file

        with open(save_dir + '/specs.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            if FLAGS.PrivAgent == True:
                writer.writerow([0]+[PrivacyAgent.get_m(r) for r in range(len(Delta_accountant)-1)])
            if FLAGS.PrivAgent == False:
                writer.writerow([0]+[FLAGS.m]*(len(Delta_accountant)-1))
            writer.writerow(Delta_accountant)
            writer.writerow(Accuracy_accountant)

    if FLAGS.relearn == True:
        # If there already is progress associated to the learned model, we do not need to store the deltas and m's as
        # they were already saved; we just keep track of the accuracy and append it to the already existing .csv file.
        # This will help us later on to average the performance, as the variance is very high.

        if len(Accuracy_accountant) > 1 or len(Accuracy_accountant) == 1 and FLAGS.loaded is True:
            # If we already appended a new line to the .csv file, we have to delete that line.
            with open(save_dir + '/specs.csv', 'r+w') as csvfile:
                csvReader = csv.reader(csvfile, delimiter=",")
                lines =[]
                for row in csvReader:
                    lines.append([float(i) for i in row])
                lines = lines[:-1]

            with open(save_dir + '/specs.csv', 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for line in lines:
                    writer.writerow(line)

        # Append a line to the .csv file holding the accuracies.
        with open(save_dir + '/specs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(Accuracy_accountant)


def global_step_creator():
    global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')
    global_step_placeholder = tf.placeholder(dtype=tf.float32, shape=(), name='global_step_placeholder')
    one = tf.constant(1, dtype=tf.float32, name='one')
    new_global_step = tf.add(global_step, one)
    increase_global_step = tf.assign(global_step, new_global_step)
    set_global_step = tf.assign(global_step, global_step_placeholder)
    return global_step, increase_global_step, set_global_step


def bring_Accountant_up_to_date(Acc, sess, rounds, PrivAgent, FLAGS):
    print('Bringing the accountant up to date....')

    for r in range(rounds):
        if FLAGS.PrivAgent:
            Sigma = PrivAgent.get_Sigma(r)
            m = PrivAgent.get_m(r)
        else:
            Sigma = FLAGS.Sigma
            m = FLAGS.m
        print('Completed '+str(r+1)+' out of '+str(rounds)+' rounds')
        t = Acc.accumulate_privacy_spending(0, Sigma, m)
        sess.run(t)
        sess.run(t)
        sess.run(t)
    print('The accountant is up to date!')

def print_loss_and_accuracy(global_loss,accuracy):
    print(' - Current Model has a loss of:           %s' % global_loss)
    print(' - The Accuracy on the validation set is: %s' % accuracy)
    print('--------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------')


def print_new_comm_round(real_round):
    print('--------------------------------------------------------------------------------------')
    print('------------------------ Communication round %s ---------------------------------------' % str(real_round))
    print('--------------------------------------------------------------------------------------')

def check_validaity_of_FLAGS(FLAGS):
    FLAGS.PrivAgent = True
    if not FLAGS.m == 0:
        if FLAGS.Sigma == 0:
            print('\n \n -------- If m is specified the Privacy Agent is not used, then Sigma has to be specified too. --------\n \n')
            raise NotImplementedError
    if not FLAGS.Sigma == 0:
        if FLAGS.m ==0:
            print('\n \n-------- If Sigma is specified the Privacy Agent is not used, then m has to be specified too. -------- \n \n')
            raise NotImplementedError
    if not FLAGS.Sigma == 0 and not FLAGS.m == 0:
        FLAGS.PrivAgent = False
    return FLAGS


def run_training(PrivacyAgent, FLAGS):

    FLAGS = check_validaity_of_FLAGS(FLAGS)

    # At this point, FLAGS.save_dir specifies both; where we save and where we assume the data is stored

    save_dir = create_save_dir(FLAGS)

    data_set, client_set, test_set = get_data(FLAGS)

    with tf.Graph().as_default():

        ################################################################################################################
        ################################################################################################################

        # - placeholder for the input Data (in our case MNIST), depends on the batch size specified in FLAGS.C
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.B)

        # - logits : output of the fully connected neural network when fed with images. The NN's architecture is
        #           specified in 'FLAGS'
        logits = mnist.inference_no_bias(images_placeholder, FLAGS)

        # - loss : when comparing logits to the true labels.
        loss = mnist.loss(logits, labels_placeholder)

        # - eval_correct: When run, returns the amount of labels that were predicted correctly.
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        ################################################################################################################
        ################################################################################################################

        # - global_step :          A Variable, which tracks the amount of steps taken by the clients:
        # - increase_global_step : When run, sets the value of global_step to global_step+1
        # - set_global_step :      When run, sets the value of global_step to that specified in a placeholder named
        #                         'global_step_placeholder'
        global_step, increase_global_step, set_global_step = global_step_creator()

        # - learning_rate : A tensorflow learning rate, dependent on the global_step variable.
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 27000, 0.1, staircase=False,
                                                   name='learning_rate')

        # - train_op : A tf.train operation, which backpropagates the loss and updates the model according to the
        #              learning rate specified.
        train_op = mnist.training(loss, learning_rate)

        ################################################################################################################
        ################################################################################################################

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
        model, Accuracy_accountant, Delta_accountant, Acc, real_round, FLAGS, Computed_deltas = load_from_directory_or_initialize(save_dir,FLAGS)

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
                if FLAGS.vali_num < len(test_set[1]):
                    # Choose FLGAS.vali_num random samples from the validation set
                    randint = np.random.randint(0, len(test_set[0]), FLAGS.vali_num)
                    feed_dict = {'images_placeholder:0': test_set[0][randint], 'labels_placeholder:0': test_set[1][randint]}
                else:
                    feed_dict = {'images_placeholder:0': test_set[0],
                                 'labels_placeholder:0': test_set[1]}

                # compute the loss on the validation set.
                global_loss = sess.run(loss, feed_dict=feed_dict)
                count = sess.run(eval_correct, feed_dict=feed_dict)
                accuracy = float(count) / float(FLAGS.vali_num)
                Accuracy_accountant.append(accuracy)

                print_loss_and_accuracy(global_loss, accuracy)

            if Delta_accountant[-1] > PrivacyAgent.get_bound() or math.isnan(Delta_accountant[-1]):
                print('************** The last step exhausted the privacy budget **************')
                if not math.isnan(Delta_accountant[-1]):
                    try: None
                    finally: save_progress(save_dir, model, Delta_accountant+[float('nan')], Accuracy_accountant+[float('nan')],PrivacyAgent, FLAGS)
                    return Accuracy_accountant, Delta_accountant, model
            else:
                try: None
                finally: save_progress(save_dir, model, Delta_accountant, Accuracy_accountant,PrivacyAgent, FLAGS)

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
            participating_clients = [client_set[k] for k in S]

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
                        feed_dict = {'images_placeholder:0': data_set[[int(j) for j in batch_ind]][:, 1:],
                                     'labels_placeholder:0': data_set[[int(j) for j in batch_ind]][:, 0]}

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
            print(' - Epsilon-Delta Privacy:'+ str([FLAGS.eps, delta]))

    return [], [], []

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training(PrivacyAgent=PrivAgent(FLAGS.N), FLAGS=FLAGS)

class flag:
    def __init__(self):
        self.PrivAgentName = 'default_Priv_Agent'
        self.GM = 1
        self.PrivAgent = True
        self.N = 100
        self.Record_privacy = True
        self.Sigma = 1.8
        self.eps = 8
        self.m = 100
        self.B = 10
        self.E = 4
        self.learning_rate = 0.1
        self.hidden1 = 600
        self.hidden2 = 100
        self.input_data_dir = '/Users/d071496/PycharmProjects/Data_for_Learning'
        self.save_dir = '/Users/d071496/PycharmProjects/Data_for_Learning'
        self.log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/fully_connected_feed')
        self.Max_comm_rounds = 3000
        self.vali_num = 10000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--GM',
        type=int,
        default=1,
        help='Wether or not a Gaussian mechanism shall be used to distort'
    )
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
        '--Record_privacy',
        type=bool,
        default=True,
        help='whether or not the Gaussian_Moments_Accountant shall be used to record delta and epsilon and stop training once they reach a critical level'
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
        '--learning_rate',
        type=float,
        default=0.1,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=600,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=100,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='DATA',
        help='Directory to put the input data.'
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
    parser.add_argument(
        '--vali_num',
        type=int,
        default=10000,
        help='number of examples to test on; 10000 is maximum; subset is chosen randomly'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
