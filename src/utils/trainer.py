import numpy as np
import os, json, torch

import matplotlib.pyplot as plt

from datetime    import datetime as dt
from collections import deque
from unityagents import UnityEnvironment

from utils import memory, utils, NN
from tqdm  import tqdm

config = json.load(open('config.json'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def saveResults(allScores):

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
    folder = f'../results/{now}'
    os.makedirs( folder )

    # ----------- Save the scores as an array ---------------
    allScores = np.array( allScores )
    np.save( os.path.join(folder, 'scores.npy'), allScores )

    # ----------- Save a plot of the scores ------------------
    plt.figure(figsize=(5,4))
    ax1 = plt.axes([0.17, 0.1, 0.66, 0.8])
    plt.plot( allScores[:,0], color='royalblue' )
    ax1.set_ylabel('mean', color='royalblue')
    ax1.tick_params('y', colors='royalblue')

    ax2 =  ax1.twinx()
    ax2.plot( allScores[:,1], color='orange' )
    ax2.set_ylabel('max', color='orange')
    ax2.tick_params('y', colors='orange')

    plt.savefig( os.path.join(folder, 'scores.png'), dpi=300)
    plt.close('all')

    # ----------- Save a copy of the current config file -----------
    with open( os.path.join(folder, 'config.json'), 'w' ) as fOut:
        fOut.write( json.dumps(config) )

    return folder

def train():

    nAgents    = 2
    agents     = [NN.Agent() for _ in range(nAgents)]
    allScores  = deque([], maxlen=100)
    allScores1 = [] # This saves the actual values

    printEvery      = config['training']['printEvery']
    totalIterations = config['training']['totalIterations']
    memorySize      = config['training']['memorySize']
    sampleSize      = config['training']['sampleSize']
    nReduce         = config['training']['nReduce']
    exploreFactor   = config['training']['initExplore']

    def explorePolicy(explore = 0):
        '''
            This taked an exploration variable and returns a policy
            that can be used. 
        '''
        def policy(states):
            
            states  = torch.from_numpy(states).float().to(device)
            actions = []
            for i, s in enumerate(states):
                actions.append( agents[i].actorSlow( s ).cpu().data.numpy().reshape(-1, 2) )

            actions       = np.vstack( actions )
            randomActions = utils.randomPolicy( states )
            actions       = (1-explore) * actions + explore * randomActions
            del states
            return actions

        return policy
    
    with utils.Env(showEnv=False, trainMode=True) as env:
        
        print('Generating memories ....')
        print('------------------------')
        for m in tqdm(range(totalIterations)):

            env.reset()

            # Update buffer should always contain some element
            # of exploration
            allResults = env.episode( explorePolicy( exploreFactor ), memorySize )

            if m % config['training']['exploreDecEvery'] == 0:
                exploreFactor *= config['training']['exploreDec']

            scores = []
            for i, result in enumerate(allResults):
                agents[i].updateBuffer(result, nReduce = nReduce)

                # Learn from a sample of 50 tuples
                agents[i].step( sampleSize )

                rewards = list(zip(*result))[2]
                scores.append( sum(rewards) )

            allScores.append( sum(rewards) )
            allScores1.append([np.mean(allScores), np.std(allScores)])

            if m%printEvery == 0:
                tqdm.write('mean = {:9.5f}, max = {:9.5f}'.format(np.mean(allScores), np.std(allScores)))

        folder = saveResults( allScores1 )
        for i, agent in enumerate(agents):
            agent.save( folder, f'Agent_{i}' )
                

    return
