from collections import deque, namedtuple
import numpy as np 

class ReplayBuffer:

    def __init__(self, maxEpisodes):
        self.maxEpisodes  = maxEpisodes
        self.memory       = deque(maxlen=maxEpisodes)
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

        state, action, reward, next_state, done = zip(*self.memory)

        reward = np.abs(reward) + epsilon # learn both bad and good
        reward = 1/reward
        prob   = reward / reward.sum()
        choice = np.random.choice( np.arange( len(self.memory) ), N, replace = False, p = prob )

        choice = sorted(list(choice), reverse=True)
        for c in choice:
            del self.memory[c]

        return

    def sample(self, nSamples, epsilon=1e-4):

        state, action, reward, next_state, done = zip(*self.memory)

        reward = np.abs(reward) + epsilon # learn both bad and good 
        prob   = reward / reward.sum()
        # choice = np.random.choice( np.arange( len(self.memory) ), nSamples, p = prob )
        choice = np.random.choice( np.arange( len(self.memory) ), nSamples )

        results = [ self.memory[c] for c in choice]
        
        return results

