from tqdm import tqdm
import numpy as np
import os, json, torch

from unityagents import UnityEnvironment
from utils import utilsTest, NN
# from utils import memory, utils, NN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def testing( folders, nTimes=5, map_location=None ):

    # Load the agents ...
    nAgents    = 2
    agents     = [NN.Agent() for _ in range(nAgents)]

    def compPolicy(states):
            
        actions = []
        for i, s in enumerate(states):
            s  = torch.from_numpy(s).float().to(device)
            actions.append( agents[i].actorSlow( s ).cpu().data.numpy().reshape(-1, 2) )

        states        = torch.from_numpy(states).float().to(device)
        actions       = np.vstack( actions )
        del states
        
        return actions

    def explorePolicy(explore = 0):
        '''
            This taked an exploration variable and returns a policy
            that can be used. 
        '''
        def policy(states):
            compActions   = compPolicy( states )
            randomActions = utilsTest.randomPolicy( states )

            if np.random.rand() <= explore:
                return randomActions
            else:
                return compActions
            
            return randomActions

        return policy

    def policy(states):
        # --------------------------------------------------------
        # The policy just takes a set of two states, and splits
        # them into two actions. Then, uses the actions as used
        # by the individual policies and returns them ...
        # --------------------------------------------------------

        states  = torch.from_numpy(states).float().to(device)
        actions = []
        for i, s in enumerate(states):
            actions.append( agents[i].actorSlow( s ).cpu().data.numpy().reshape(-1, 2) )
        del states

        actions = np.vstack( actions )
        return actions

    summary_r1 = np.zeros((6, 6, nTimes))
    summary_r2 = np.zeros((6, 6, nTimes))

    d1 = [0.6]
    with utilsTest.Env(showEnv=True, trainMode=False) as env:

        for m1, x1 in enumerate(tqdm(d1)):
            for m2, x2 in enumerate(tqdm(d1)):

                for folder in folders:
                    tqdm.write(f'Loading data from folder : [{folder}]')
                    for i, agent in enumerate(agents):
                        # agent.load( folder, f'Agent_{i}', map_location = map_location )
                        agent.load( folder, f'Agent_0', map_location = map_location )

                    for i in tqdm(range(nTimes)):
                        env.reset()
                        allResults = env.episode( explorePolicy( x1 ), 100 , explorePolicy( x2 ))
                        r1, r2 = sum(list(zip(*allResults[0]))[2]) , sum(list(zip(*allResults[1]))[2]) 
                        if (r1 > 0.01) and (r2 > 0.09):
                            tqdm.write(str((r1, r2)))

                        summary_r1[m1, m2, i] = r1
                        summary_r2[m1, m2, i] = r2


    # print(summary_r1.max(axis=2))
    # print(summary_r1.mean(axis=2))
    # print(summary_r1.std(axis=2))

    # print(summary_r2.max(axis=2))
    # print(summary_r2.mean(axis=2))
    # print(summary_r2.std(axis=2))

    # np.save('../results/r1.npy', summary_r1)
    # np.save('../results/r2.npy', summary_r2)


    return