import numpy as np
import os, json, torch

import matplotlib.pyplot as plt

from datetime    import datetime as dt
from collections import deque

from utils import utils, NN, generateMemories
from tqdm  import tqdm

config = json.load(open('config.json'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

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
    ax1.set_ylabel('max', color='royalblue')
    ax1.tick_params('y', colors='royalblue')

    ax2 =  ax1.twinx()
    ax2.plot( allScores[:,1], color='orange' )
    ax2.set_ylabel('mean', color='orange')
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
    allQLoss, allALoss = [], []

    printEvery      = config['training']['printEvery']
    totalIterations = config['training']['totalIterations']
    nSteps          = config['training']['nSteps']
    memorySize      = config['training']['memorySize']
    sampleSize      = config['training']['sampleSize']
    exploreFactor   = config['training']['initExplore']
    hotStart        = config['training']['hotStart']

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
            randomActions = utils.randomPolicy( states )

            if np.random.rand() <= explore:
                return randomActions
            else:
                return compActions
            
            return randomActions

        return policy
    
    with utils.Env(showEnv=False, trainMode=True) as env:
        



        if hotStart is None:
            print('Generating memories ....')
            print('------------------------')
            
            allResults = generateMemories.memories( env, 10000, explorePolicy( exploreFactor ), episodeSize = memorySize )
            for i, result in enumerate(allResults):
                agents[i].updateBuffer(result)
        else:
            print(f'Hot-starting agents from a previous state [{hotStart}]')
            print('---------------------------------------------------------------------------')
            for i in range(2):
                agents[i].load(hotStart, f'Agent_{i}')

        for m in tqdm(range(totalIterations)):

            env.reset()

            # Update buffer should always contain some element
            # of exploration
            allResults = generateMemories.memories( env, 10, explorePolicy( exploreFactor ), episodeSize = memorySize )

            if m % config['training']['exploreDecEvery'] == 0:
                exploreFactor *= config['training']['exploreDec']

            for i, result in enumerate(allResults):
                agents[i].updateBuffer(result)

                # Learn from a sample of ``sampleSize`` tuples ``nSteps`` times
                loss = [agents[i].step( sampleSize ) for _ in range(nSteps) ]
                qLoss, aLoss = zip(*loss)
                allQLoss.append( np.hstack(qLoss).mean() )
                allALoss.append( np.hstack(aLoss).mean() )


                agents[i].softUpdate()

            # Here, we need to play a set of episoodes to find the scores
            # allResults = env.episode(compPolicy, maxSteps = 5000)
            allResults = env.episode(compPolicy, maxSteps = 5000)
            rewards1 = np.array(list(zip(*allResults[0]))[2]) 
            rewards2 = np.array(list(zip(*allResults[1]))[2]) 

            allScores.append( max(rewards1.sum(), rewards2.sum()) )
            allScores1.append([np.sum(rewards1), np.mean(rewards1), np.std(rewards1)])

            if m%printEvery == 0:
                tqdm.write('mean = {:9.5f}, max = {:9.5f}, explore = {}, qLoss = {}, aLoss = {}'.format(
                    np.mean(allScores), np.std(allScores), exploreFactor,
                    np.array(allQLoss).mean(), np.array(allALoss).mean() ))

        folder = saveResults( allScores1 )
        for i, agent in enumerate(agents):
            agent.save( folder, f'Agent_{i}' )
                

    return
