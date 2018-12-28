import numpy as np
import os, json, torch

from unityagents import UnityEnvironment
from utils import utils, NN
# from utils import memory, utils, NN

config = json.load(open('config.json'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def testing( folders, nTimes=5, map_location=None ):

    # Load the agents ...
    nAgents    = 2
    agents     = [NN.Agent() for _ in range(nAgents)]

    def policy(states):
        # --------------------------------------------------------
        # The policy just takes a set of two states, and splits
        # them into two actions. Then, uses the actions as used
        # by the individual policies and returns them ...
        # --------------------------------------------------------

        actions = []
        for i, s in enumerate(states):
            if len(s.shape) == 1:
                s = s.reshape((1, -1))
            s  = torch.from_numpy(s).float().to(device)
            actions.append( agents[i].actorSlow( s ).cpu().data.numpy().reshape(-1, 2) )

        states        = torch.from_numpy(states).float().to(device)
        actions       = np.vstack( actions )
        del states
        return actions

    with utils.Env(showEnv=True, trainMode=False) as env:

        maxScores = []
        for folder in folders:
            print(f'Loading data from folder : [{folder}]')
            for i, agent in enumerate(agents):
                agent.load( folder, f'Agent_{i}', map_location = map_location )

            for i in range(nTimes):
                env.reset()
                allResults = env.episode( policy, 5000 )
                scores1 = sum(list(zip(*allResults[0]))[2])
                scores2 = sum(list(zip(*allResults[1]))[2])
                print(max(scores1, scores2))
                maxScores.append(max(scores1, scores2))

            print('Average score for this: {}'.format(np.mean(maxScores)))

    return