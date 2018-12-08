from collections import deque, namedtuple
import numpy as np 

class Episode:

    def __init__(self, maxLen, nSamples, nSteps=1, gamma=1):

        self.maxLen   = maxLen
        self.nSamples = nSamples
        self.nSteps   = nSteps
        self.gamma    = gamma 
        # We want to save the cumulative reward so that we are able
        # to do some form of value-based implementation if necessary
        self.Experience = namedtuple("Experience", 
            field_names = [ "state", "action", "reward", "next_state", 
                            "done", "cumReward"])
        self.memory = deque(maxlen=maxLen)

        return
        
    def append(self, state, action, reward, next_state, done):
        # Notice that every time an experience is added, we
        # need to update the value of cumulative rewards. So
        # We shall save them as an array at the end ...

        for m in self.memory:
            m.cumReward.append((reward, next_state))

        e = self.Experience(state, action, reward, next_state, done, [])
        self.memory.append(e)

        return

    def sample(self, nSamples=None, nSteps=None):


        if nSamples is None:
            nSamples = self.nSamples

        if nSteps is None:
            nSteps = self.nSteps

        retVals = np.random.choice( range(len(self.memory)), nSamples )
        results = []
        

        return results
