import pickle
import numpy as np
import os

def create_clients(num, dir):

    '''
    This function creates clients that hold non-iid MNIST data accroding to the experiments in https://research.google.com/pubs/pub44822.html. (it actually just creates indices that point to data.
    but the way these indices are grouped, they create a non-iid client.)
    :param num: Number of clients
    :param dir: where to store
    :return: _
    '''

    num_examples = 50000
    num_classes = 10
    if os.path.exists(dir + '/'+str(num)+'_clients.pkl'):
        print('Client exists at: '+dir + '/'+str(num)+'_clients.pkl')
        return
    if not os.path.exists(dir):
        os.makedirs(dir)
    buckets = []
    for k in range(num_classes):
        temp = []
        for j in range(num / 100):
            temp = np.hstack((temp, k * num_examples/10 + np.random.permutation(num_examples/10)))
        buckets = np.hstack((buckets, temp))
    shards = 2 * num
    perm = np.random.permutation(shards)
    # z will be of length 250 and each element represents a client.
    z = []
    ind_list = np.split(buckets, shards)
    for j in range(0, shards, 2):
        # each entry of z is associated to two shards. the two shards are sampled randomly by using the permutation matrix
        # perm and stacking two shards together using vstack. Each client now holds 250*2 datapoints.
        z.append(np.hstack((ind_list[int(perm[j])], ind_list[int(perm[j + 1])])))
        # shuffle the data in each element of z, so that each client doesn't have all digits stuck together.
        perm_2 = np.random.permutation(2 * len(buckets) / shards)
        z[-1] = z[-1][perm_2]
    filehandler = open(dir + '/'+str(num)+'_clients.pkl', "wb")
    pickle.dump(z, filehandler)
    filehandler.close()
    print('client created at: '+dir + '/'+str(num)+'_clients.pkl')

if __name__ == '__main__':
    List_of_clients = [100,200,500,1000,2000,5000,10000]
    for j in List_of_clients:
        create_clients(j, os.getcwd()+'/DATA/clients')
