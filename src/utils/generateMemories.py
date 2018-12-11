from tqdm  import tqdm
from utils import utils
import numpy as np


def memories(nIterations, policy, episodeSize, gamma = 0.8, filterVal = 0.03):

    with utils.Env(showEnv=False, trainMode=True) as env:
        
        print('Generating memories ....')
        print('------------------------')

        memories = [[] for i in range(2)]
        for m in tqdm(range(nIterations)):

            env.reset()

            results = env.episode( policy, episodeSize )
            for i, result in enumerate(results):

                state, action, reward, next_state, done = zip(*result)
                reward = np.array(reward)
                
                if np.any( np.array(reward) > 0.09 ):

                    cumReward = [ np.sum(reward[i+1:] * (gamma ** np.arange(len(reward)-i-1))) for i in range(len(reward)-1)]
                    cumReward = np.array( cumReward )
                    mask      = cumReward > filterVal
                    maskNums  = np.arange(len(mask))[ mask ]


                    # Generate the masked data ...
                    state      = [state[m]      for m in maskNums]
                    action     = [action[m]     for m in maskNums]
                    reward     = [reward[m]     for m in maskNums]
                    next_state = [next_state[m] for m in maskNums]
                    done       = [done[m]       for m in maskNums]
                    cumReward  = [cumReward[m]  for m in maskNums]

                    memories[i].append(list(zip( 
                        state, action, reward, next_state, done, cumReward )))

    return memories


