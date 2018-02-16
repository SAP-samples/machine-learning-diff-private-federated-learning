
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import csv
import os.path
import pickle
from accountant import GaussianMomentsAccountant
import math
import os
import tensorflow as tf

class PrivAgent:
    def __init__(self, N, Name):
        self.N = N
        self.Name = Name
        if N == 100:
            self.m = [30]*100
            self.Sigma = [1]*24
            self.bound = 0.001
        if N == 1000:
            self.m = [100]*10
            self.Sigma = [1]*24
            self.bound = 0.00001
        if N == 10000:
            self.m = [300]*10
            self.Sigma = [1]*24
            self.bound = 0.000001
        if(N != 100 and N != 1000 and N != 10000 ):
            print('!!!!!!! YOU CAN ONLY USE THE PRIVACY AGENT FOR N = 100, 1000 or 10000 !!!!!!!')

    def get_m(self, r):
        return self.m[r]

    def get_Sigma(self, r):
        return self.Sigma[r]

    def get_bound(self):
        return self.bound

def Assignements(dic):
    return [tf.assign(var, dic[Vname_to_Pname(var)]) for var in tf.trainable_variables()]


def Vname_to_Pname(var):
    return var.name[:var.name.find(':')] + '_placeholder'


def Vname_to_FeedPname(var):
    return var.name[:var.name.find(':')] + '_placeholder:0'


def Vname_to_Vname(var):
    return var.name[:var.name.find(':')]


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
        if FLAGS.record_privacy == True:
            if FLAGS.relearn == False:
                # I.e. we never learned a complete model before and have therefore never computed all deltas.
                for j in range(len(self.keys)):
                    sess.run(t)
                r = Acc.get_privacy_spent(sess, [FLAGS.eps])
                delta = r[0][1]
            else:
                # I.e. we have computed a complete model before and can reuse the deltas from that time.
                delta = Computed_deltas[self.round]
        return New_model, delta


def create_save_dir(FLAGS):
    '''
    :return: Returns a path that is used to store training progress; the path also identifies the chosen setup uniquely.
    '''
    raw_directory = FLAGS.save_dir + '/'
    if FLAGS.gm: gm_str = 'Dp/'
    else: gm_str = 'non_Dp/'
    if FLAGS.priv_agent:
        model = gm_str + 'N_' + str(FLAGS.n) + '/Epochs_' + str(
            int(FLAGS.e)) + '_Batches_' + str(int(FLAGS.b))
        return raw_directory + str(model) + '/' + FLAGS.PrivAgentName
    else:
        model = gm_str + 'N_' + str(FLAGS.n) + '/Sigma_' + str(FLAGS.Sigma) + '_C_'+str(FLAGS.m)+'/Epochs_' + str(
            int(FLAGS.e)) + '_Batches_' + str(int(FLAGS.b))
        return raw_directory + str(model)


def load_from_directory_or_initialize(directory, FLAGS):
    '''
    This function looks for a model that corresponds to the characteristics specified and loads potential progress.
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
    Acc = GaussianMomentsAccountant(FLAGS.n)
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
            if FLAGS.priv_agent == True:
                writer.writerow([0]+[PrivacyAgent.get_m(r) for r in range(len(Delta_accountant)-1)])
            if FLAGS.priv_agent == False:
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
    global_step = [v for v in tf.global_variables() if v.name == "global_step:0"][0]
    global_step_placeholder = tf.placeholder(dtype=tf.float32, shape=(), name='global_step_placeholder')
    one = tf.constant(1, dtype=tf.float32, name='one')
    new_global_step = tf.add(global_step, one)
    increase_global_step = tf.assign(global_step, new_global_step)
    set_global_step = tf.assign(global_step, global_step_placeholder)
    return increase_global_step, set_global_step


def bring_Accountant_up_to_date(Acc, sess, rounds, PrivAgent, FLAGS):
    '''

    :param Acc: A Privacy accountant
    :param sess: A tensorflow session
    :param rounds: the number of rounds that the privacy accountant shall iterate
    :param PrivAgent: A Privacy_agent that has functions: PrivAgent.get_Sigma(round) and PrivAgent.get_m(round)
    :param FLAGS: priv_agent specifies whether to use a PrivAgent or not.
    :return:
    '''
    print('Bringing the accountant up to date....')

    for r in range(rounds):
        if FLAGS.priv_agent:
            Sigma = PrivAgent.get_Sigma(r)
            m = PrivAgent.get_m(r)
        else:
            Sigma = FLAGS.sigma
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
    FLAGS.priv_agent = True
    if not FLAGS.m == 0:
        if FLAGS.sigma == 0:
            print('\n \n -------- If m is specified the Privacy Agent is not used, then Sigma has to be specified too. --------\n \n')
            raise NotImplementedError
    if not FLAGS.sigma == 0:
        if FLAGS.m ==0:
            print('\n \n-------- If Sigma is specified the Privacy Agent is not used, then m has to be specified too. -------- \n \n')
            raise NotImplementedError
    if not FLAGS.sigma == 0 and not FLAGS.m == 0:
        FLAGS.priv_agent = False
    return FLAGS

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