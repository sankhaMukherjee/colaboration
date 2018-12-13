from tqdm  import tqdm
from utils import utils
import numpy as np


def memories(env, nIterations, policy, episodeSize, gamma = 0.8, filterVal = 0.03):
    '''generate a set of memories to append it to the data
    
    This function takes a Unity Environment and plays with it, uisng the supplied
    policy. Since this has two agents playing the game at the same time, the 
    function separates the results into two separate sets of memories, one for
    each agent. This allows each of the agents to load its own memories, rather
    than saving both memories within the same state.

    Furthermore, it prefers to only save memories that will yield some form of
    cumulative reward. This is possible because the task is episodic, and there
    is a very high chance of gaining a point. Furthermore, typically, gaining
    one point is independent of gaining another point. Hence, for each episode that
    the agent plays, this will save the ones where there is some positive cumulative 
    rewards associated with the move. 

    The value of ``gamma`` and ``filterVal`` have been chosen such that the future
    reward is not influenced by a previous point gain. Hence, if an agent is able
    to hit the ball twice, the cumulative gain resulting form the second point will
    decay before it reaches the first point. 
    
    Parameters
    ----------
    env : {Unity.Environment object}
        This is Tennis Unity environment
    nIterations : {int}
        Number of episodes that the agents should play so that we are able to
        generate the right episode. 
    policy : {[type]}
        [description]
    episodeSize : {[type]}
        [description]
    gamma : {number}, optional
        [description] (the default is 0.8, which [default_description])
    filterVal : {number}, optional
        [description] (the default is 0.03, which [default_description])
    
    Returns
    -------
    list
        This is a list of two lists. The first one contains a list of tuples which happens to
        be the data for the first actor, and the second one is that for the second actor.
    '''
        
    memories = [[] for i in range(2)]
    for m in tqdm(range(nIterations)):

        env.reset()

        results = env.episode( policy, episodeSize )
        for i, result in enumerate(results):

            state, action, reward, next_state, done = zip(*result)
            reward    = np.array(reward)
            cumReward = [ np.sum(reward[i+1:] * (gamma ** np.arange(len(reward)-i-1))) for i in range(len(reward)-1)]
            cumReward = np.array( cumReward )

            # Save the cumulative rewards without decay. This will allow the 
            # rewards to be plotted properly
            cumReward1 = [ np.sum(reward[i+1:]) for i in range(len(reward)-1)]
            cumReward1 = np.array( cumReward1 )
            
            if np.any( cumReward > filterVal ):
                mask      = cumReward > filterVal
                maskNums  = np.arange(len(mask))[ mask ]

                # Generate the masked data ...
                for n in maskNums:
                    tup = state[n] , action[n], reward[n], next_state[n], done[n], cumReward1[n]
                    memories[i].append(tup)

    return memories


