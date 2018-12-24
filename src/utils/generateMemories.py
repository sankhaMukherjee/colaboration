from tqdm  import tqdm
from utils import utils
import numpy as np


def memories(env, nIterations, policy, episodeSize, gamma = 0.8, filterVal = 0.03, minScoreAdd=0.09, propBad=0.03, policy1=None, t1=14):
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
    policy : {function}
        This takes a state and returns the action for each agent.
    episodeSize : {integer}
        The total iteration size that one episode will have to go through.
    gamma : real, optional
        The value of the discount factor that will be used for the calculation of how close 
        a timepoint will have to be for insertion into the ReplayBuffer (the default is 0.8)
    filterVal : real, optional
        values of cumulative sum below this are considered for insertion into the ReplayBuffer (
        the default is 0.03)
    propBad : {number}, optional
        proportional of bad values that this should be inserted into the data buffer (the default 
        is 0.03, which results in about 3% of the times bad data is encounteres, it will be inserted
        into the ReplayBuffer)
    
    Returns
    -------
    list
        This is a list of two lists. The first one contains a list of tuples which happens to
        be the data for the first actor, and the second one is that for the second actor.
    '''
        
    memories = [[] for i in range(2)]
    for m in range(nIterations):

        if m % 50 == 0:
            print('\r{} complete                               '.format( m*100.0/nIterations ), end='', flush=True)

        env.reset()

        if policy1 is None:
            results = env.episode( policy, episodeSize )
        else:
            results = env.episode( policy, episodeSize, policy1, t1 )

        for i, result in enumerate(results):

            state, action, reward, next_state, done = zip(*result)
            reward    = np.array(reward)
            cumReward = [ np.sum(reward[j+1:] * (gamma ** np.arange(len(reward)-j-1))) for j in range(len(reward)-1)]
            cumReward = np.array( cumReward )

            totalHits = []
            for j in range(len(reward)):
                before = sum(reward[:j] > 0.09)
                after  = np.any( reward[j:] > 0.09 )*1
                totalHits.append((before+after)*after)
                
            # Save the cumulative rewards without decay. This will allow the 
            # rewards to be plotted properly
            cumReward1 = [ np.sum(reward[j+1:]) for j in range(len(reward)-1)]
            cumReward1 = np.array( cumReward1 )
            
            if np.any( cumReward > filterVal ):
                if reward.sum() > 0.11:
                    tqdm.write('maxScore for Agent {} : {}'.format( i, reward.sum() ))
                    # tqdm.write('rewards    {} : {}'.format( i, reward ))
                    # tqdm.write('Total Hits {} : {}'.format( i, totalHits ))
                    # tqdm.write('Cumrewards {} : {}'.format( i, cumReward ))
                    # tqdm.write('Cumreward1 {} : {}'.format( i, cumReward1 ))
                
                mask      = cumReward > filterVal
                maskNums  = np.arange(len(mask))[ mask ]

                if reward.sum() > minScoreAdd:
                    # We dont want to flood memories with single hits
                    # Generate the masked data ...
                    for n in maskNums:
                        tup = state[n] , action[n], reward[n], next_state[n], done[n], cumReward1[n], totalHits[n]
                        memories[0].append(tup)
                        memories[1].append(tup)

            elif np.random.rand() <= propBad:

                for n in range( len(state) ):
                        tup = state[n] , action[n], reward[n], next_state[n], done[n], -0.001, 0
                        memories[0].append(tup)
                        memories[1].append(tup)

            else:
                pass

    return memories


