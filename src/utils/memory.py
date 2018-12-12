from collections import deque, namedtuple
import numpy as np
import pickle, os

class ReplayBuffer:

    def __init__(self, maxDataTuples):
        '''The replay buffer
        
        Save data for the replay buffer
        
        Parameters
        ----------
        maxDataTuples : {int}
            The size of the ``deque`` that is used for storing the
            data tuples. This assumes that the data tuples are 
            present in the form: ``(state, action, reward, next_state, 
            done, cumRewards)``. This means that we assume that the 
            data will have some form of cumulative reward pints associated
            with each tuple.
        '''
        self.maxDataTuples  = maxDataTuples
        self.memory       = deque(maxlen=maxDataTuples)
        return

    def append(self, result):
        self.memory.append(result)
        return

    def appendMany(self, results):
        for r in results:
            self.memory.append(r)
        return

    def delNVals(self, N, epsilon=1e-4):

        if N*3 >= len(self.memory):
            return

        state, action, reward, next_state, done, cumRewards = zip(*self.memory)

        reward = np.abs(reward) + epsilon # learn both bad and good
        reward = 1/reward
        prob   = reward / reward.sum()
        choice = np.random.choice( np.arange( len(self.memory) ), N, replace = False, p = prob )

        choice = sorted(list(choice), reverse=True)
        for c in choice:
            del self.memory[c]

        return

    def sample(self, nSamples, epsilon=1e-4):

        result = zip(*self.memory)
        state, action, reward, next_state, done, cumRewards = result


        reward = np.abs(reward) + epsilon # learn both bad and good 
        prob   = reward / reward.sum()
        choice = np.random.choice( np.arange( len(self.memory) ), nSamples, p = prob )
        # choice = np.random.choice( np.arange( len(self.memory) ), nSamples )

        results = [ self.memory[c] for c in choice]
        
        return results

    def save(self, folder, name):

        with open(os.path.join(folder, f'memory_{name}.pickle'), 'wb') as fOut:
            pickle.dump(self.memory, fOut, pickle.HIGHEST_PROTOCOL)

        return

    def load(self, folder, name):
        self.memory = pickle.load(os.path.join(folder, f'memory_{name}.pickle'))
        return

