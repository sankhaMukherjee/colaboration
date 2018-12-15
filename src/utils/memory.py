from collections import deque
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
        '''append a single tuple to the current replay buffer
        
        This function allows someone to add a single tuple to
        the replay buffer. 
        
        Parameters
        ----------
        result : {tuple}
            The tuple that should be added into the memory buffer. 
        '''
        self.memory.append(result)
        return

    def appendMany(self, results):
        '''append multiple tuples to the memory buffer
        
        Most often we will not be insterested in inserting a single data point
        into the replay buffer, but rather a whole list of these. This function
        just iterates over this list and inserts each tuple one by one.
        
        Parameters
        ----------
        results : {list}
            List of tuples that are to be inserted into the replay buffer.
        '''
        for r in results:
            self.memory.append(r)
        return

    def delNVals(self, N, epsilon=1e-4):
        '''dont use this function. 
        
        [description]
        
        Parameters
        ----------
        N : {[type]}
            [description]
        epsilon : {number}, optional
            [description] (the default is 1e-4, which [default_description])
        '''

        if N*3 >= len(self.memory):
            return

        state, action, reward, next_state, done, cumRewards, totalHits = zip(*self.memory)

        reward = np.abs(reward) + epsilon # learn both bad and good
        reward = 1/reward
        prob   = reward / reward.sum()
        choice = np.random.choice( np.arange( len(self.memory) ), N, replace = False, p = prob )

        choice = sorted(list(choice), reverse=True)
        for c in choice:
            del self.memory[c]

        return

    def sample(self, nSamples, epsilon=1e-4):
        '''sample from the replay beffer
        
        This function samples form the memory buffer, and returns the number of
        samples required. This does sampling in an intelligent manner. Since we are
        saving the cumulative rewards, we selectively select values that provide
        us greater 
        
        Parameters
        ----------
        nSamples : {[type]}
            [description]
        epsilon : {number}, optional
            [description] (the default is 1e-4, which [default_description])
        
        Returns
        -------
        list
            A list of samples that can be used for sampling the data. 
        '''

        result = zip(*self.memory)
        state, action, reward, next_state, done, cumRewards, totalHits = result

        x    = np.array(cumRewards) + np.array(totalHits)
        x    = x + epsilon
        prob = x / x.sum()

        choice = np.random.choice( np.arange( len(self.memory) ), nSamples, p = prob )
        # choice = np.random.choice( np.arange( len(self.memory) ), nSamples )

        results = [ self.memory[c] for c in choice]
        
        return results

    def save(self, folder, name):
        '''save the replay buffer
        
        This function is going to save the data within the replay buffer
        into a pickle file. This will allow us to reload the buffer to 
        a state where it has already been saved.
        
        Parameters
        ----------
        folder : {str}
            path to the folder where the data is to be saved
        name : {str}
            Name associated with the buffer. Since this program has two agents
            acting in tandum, we need to provide a name that will identify which
            agent's buffer we are saving. 
        '''

        with open(os.path.join(folder, f'memory_{name}.pickle'), 'wb') as fOut:
            pickle.dump(self.memory, fOut, pickle.HIGHEST_PROTOCOL)

        return

    def load(self, folder, name):
        '''load the data from a particular file
        
        Data saved with the previous command can be reloaded into this new buffer.
        
        Parameters
        ----------
        folder : {str}
            Path to the folder where the data is saved
        name : {str}
            Name of the agent associated whose data is to be extracted.
        '''
        self.memory = pickle.load(open( os.path.join(folder, f'memory_{name}.pickle'), 'rb' ))
        return

