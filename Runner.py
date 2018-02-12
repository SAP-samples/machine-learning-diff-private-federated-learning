from Redone import run_training, flag
import numpy as np

class PrivAgent:
    def __init__(self, N):
        self.N = N
        if N == 100:
            self.m = [20,30,40,45,45,45,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
            self.Sigma = [1]*24
            self.bound = 0.001
        if N == 1000:
            self.m = [20,30,40,45,45,45,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
            self.Sigma = [1]*24
            self.bound = 0.00001
        if N == 10000:
            self.m = [200, 200, 200, 200]
            self.Sigma = [1]*24
            self.bound = 0.000001
        if(N != 100 and N != 1000 and
          N != 10000 ):
            print('!!!!!!! YOU CAN ONLY USE THE PRIVACY AGENT FOR N = 100, 1000 or 10000 !!!!!!!')

    def get_m(self, r):
        return self.m[r]

    def get_Sigma(self, r):
        return self.Sigma[r]

    def get_bound(self):
        return self.bound

FLAGS = flag()
FLAGS.N = 100

PrivacyAgent = []

for j in range(10):
    PrivacyAgent.append(PrivAgent(FLAGS.N))

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

for i in range(10):
    Acc_temp = []
    Del_temp = []
    for j in range(10):
        FLAGS.PrivAgentName = 'Priv_Agent'+str(i)
        Accuracy_accountant, Delta_accountant, model = run_training(PrivacyAgent[i], FLAGS=FLAGS)
        Acc_temp.append(Accuracy_accountant)
        Del_temp.append(Delta_accountant)
Acc_temp = np.asarray(Acc_temp)
Acc_temp = np.mean(Acc_temp,0)
Acc.append(Acc_temp)
Del.append(Del_temp)

print(Acc)
print(Del)