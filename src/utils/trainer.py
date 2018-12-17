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

def saveResults(allScores, allScores_1, qLoss, aLoss, qLoss1, aLoss1):

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
    folder = f'../results/{now}'
    os.makedirs( folder )

    # ----------- Save the scores as an array ---------------
    allScores = np.array( allScores )
    allScores_1 = np.array( allScores )
    np.save( os.path.join(folder, 'scoresAgent_0.npy'), allScores )
    np.save( os.path.join(folder, 'scoresAgent_1.npy'), allScores_1 )

    # ----------- Save a plot of the scores ------------------
    plt.figure(figsize=(5,4))
    ax1 = plt.axes([0.17, 0.1, 0.66, 0.8])
    ax1.plot( allScores[:,1], 's', mfc='orange', mec='None' )
    ax1.set_ylabel('mean', color='orange')
    ax1.tick_params('y', colors='orange')

    ax2 =  ax1.twinx()
    plt.plot( allScores[:,0], 'o', mfc='royalblue', mec='None' )
    ax2.set_ylabel('max', color='royalblue')
    ax2.tick_params('y', colors='royalblue')

    plt.savefig( os.path.join(folder, 'scoresAgent_0.png'), dpi=300)
    plt.close('all')

    # ----------- Save a plot of the scores ------------------
    plt.figure(figsize=(5,4))
    ax1 = plt.axes([0.17, 0.1, 0.66, 0.8])
    plt.plot( allScores_1[:,0], color='royalblue' )
    ax1.set_ylabel('max', color='royalblue')
    ax1.tick_params('y', colors='royalblue')

    ax2 =  ax1.twinx()
    ax2.plot( allScores_1[:,1], color='orange' )
    ax2.set_ylabel('mean', color='orange')
    ax2.tick_params('y', colors='orange')

    plt.savefig( os.path.join(folder, 'scoresAgent_1.png'), dpi=300)
    plt.close('all')

    # ----------- Save a plot of the losses ------------------
    plt.figure(figsize=(5,4))
    ax1 = plt.axes([0.17, 0.1, 0.66, 0.8])
    plt.plot( qLoss, color='royalblue' )
    plt.plot( qLoss1, color='blue' )
    ax1.set_ylabel('critic loss function', color='royalblue')
    ax1.tick_params('y', colors='royalblue')
    ax1.set_yscale('log')

    ax2 =  ax1.twinx()
    ax2.plot( aLoss, color='orange' )
    ax2.plot( aLoss1, color='brown' )
    ax2.set_ylabel('actor loss function', color='orange')
    ax2.tick_params('y', colors='orange')
    # ax2.set_yscale('log')

    plt.savefig( os.path.join(folder, 'losses.png'), dpi=300)
    plt.close('all')

    # ----------- Save a copy of the current config file -----------
    with open( os.path.join(folder, 'config.json'), 'w' ) as fOut:
        fOut.write( json.dumps(config) )

    return folder

def train():

    nAgents    = 2
    agents     = [NN.Agent() for _ in range(nAgents)]
    allScores  = deque([], maxlen=100)
    allScores_0 = [] # This saves the actual values
    allScores_1 = [] # This saves the actual values
    allQLoss, allALoss = [[], []], [[], []]

    printEvery       = config['training']['printEvery']
    totalIterations  = config['training']['totalIterations']
    nSteps           = config['training']['nSteps']
    
    sampleSize       = config['training']['sampleSize']
    hotStart         = config['training']['hotStart']
    
    # Replay buffer stuff
    episodeSize      = config['training']['episodeSize']
    exploreFactor    = config['training']['initExplore']
    fillReplayBuffer = config['training']['fillReplayBuffer']
    filterVal        = config['training']['filterVal']
    propBad          = config['training']['propBad']
    minScoreAdd      = config['training']['minScoreAdd']

    prevScore = -100

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
            
            allResults = generateMemories.memories( env, fillReplayBuffer, 
                explorePolicy( exploreFactor ), 
                episodeSize = episodeSize,
                filterVal   = filterVal,
                propBad     = propBad,
                minScoreAdd = minScoreAdd
                 )
            for i, result in enumerate(allResults):
                agents[i].updateBuffer(result)
        else:
            print(f'Hot-starting agents from a previous state [{hotStart}]')
            print('---------------------------------------------------------------------------')
            for i in range(2):
                agents[i].load(hotStart, f'Agent_{i}')

            print('Generating memories ....')
            print('------------------------')
            
            allResults = generateMemories.memories( env, fillReplayBuffer, 
                explorePolicy( exploreFactor ), 
                episodeSize = episodeSize,
                filterVal   = filterVal,
                propBad     = propBad,
                minScoreAdd = minScoreAdd
                )
            for i, result in enumerate(allResults):
                agents[i].updateBuffer(result)

        for m in tqdm(range(totalIterations)):

            env.reset()

            # Update buffer should always contain some element
            # of exploration
            allResults = generateMemories.memories( env, 10, 
                explorePolicy( 0 ), 
                episodeSize = episodeSize,
                filterVal   = filterVal,
                propBad     = propBad,
                minScoreAdd = minScoreAdd )

            if m % config['training']['exploreDecEvery'] == 0:
                exploreFactor *= config['training']['exploreDec']

            for i, result in enumerate(allResults):
                agents[i].updateBuffer(result)

                # Learn from a sample of ``sampleSize`` tuples ``nSteps`` times
                loss = [agents[i].step( sampleSize ) for _ in range(nSteps) ]
                agents[i].softUpdate()

                qLoss, aLoss = zip(*loss)
                allQLoss[i].append( np.hstack(qLoss).mean() )
                allALoss[i].append( np.hstack(aLoss).mean() )


            # Here, we need to play a set of episoodes to find the scores
            allResults = env.episode(compPolicy, maxSteps = 5000)
            rewards1 = np.array(list(zip(*allResults[0]))[2]) 
            rewards2 = np.array(list(zip(*allResults[1]))[2]) 

            allScores.append( max(rewards1.sum(), rewards2.sum()) )
            allScores_0.append([np.sum(rewards1), np.mean(rewards1), np.std(rewards1)])
            allScores_1.append([np.sum(rewards2), np.mean(rewards2), np.std(rewards2)])

            # We should save the agent at every step ... 
            # -------------------------------------------
            if prevScore <= allScores[-1]:
                prevScore = allScores[-1]

                folder = f'../results/tmp/[{m}]-[{prevScore}]'
                os.makedirs(folder)

                for i, agent in enumerate(agents):
                    agent.save(folder, f'Agent_{i}')

            if m%printEvery == 0:
                tqdm.write('mean = {:9.5f}, max = {:9.5f}, explore = {}, qLoss = {}, aLoss = {}'.format(
                    np.mean(allScores), np.max(allScores), exploreFactor,
                    np.array(allQLoss[0])[-1], np.array(allALoss[0])[-1] ),
                )

        folder = saveResults( allScores_0, allScores_1, allQLoss[0], allALoss[0], allQLoss[1], allALoss[1] )
        for i, agent in enumerate(agents):
            agent.save( folder, f'Agent_{i}' )
                

    return
