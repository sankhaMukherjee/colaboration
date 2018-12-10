import numpy as np
import os, json, torch

from unityagents import UnityEnvironment
from utils import utils, NN
# from utils import memory, utils, NN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def testing( folders, nTimes=5 ):

    # Load the agents ...
    nAgents    = 2
    agents     = [NN.Agent() for _ in range(nAgents)]

    def policy(states):
        # --------------------------------------------------------
        # The policy just takes a set of two states, and splits
        # them into two actions. Then, uses the actions as used
        # by the individual policies and returns them ...
        # --------------------------------------------------------

        states  = torch.from_numpy(states).float().to(device)
        actions = []
        for i, s in enumerate(states):
            actions.append( agents[i].actorFast( s ).cpu().data.numpy().reshape(-1, 2) )
        del states

        actions = np.vstack( actions )
        return actions

    with utils.Env(showEnv=True, trainMode=False) as env:

        for folder in folders:
            print(f'Loading data from folder : [{folder}]')
            for i, agent in enumerate(agents):
                agent.load( folder, f'Agent_{i}' )

            for i in range(nTimes):
                env.reset()
                allResults = env.episode( policy, 100 )


    return