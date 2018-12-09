import json
import numpy as np

from unityagents import UnityEnvironment

from utils import memory
from utils import utils
from utils import NN
from tqdm import tqdm

config = json.load(open('config.json'))

def fillReplayBuffer():

    print('Lets fill up the replay buffer ...')
    memories = [memory.ReplayBuffer(200) for _ in range(2)]

    with utils.Env(showEnv=False, trainMode=True) as env:
        for i in tqdm(range(500)):
            env.reset()
            allResults = env.episode( utils.randomPolicy, 100 )
            for i, result in enumerate(allResults):
                memories[i].delNVals( 2 )
                memories[i].appendMany(result)


    state, action, reward, next_state, done = memories[0].memory[1]
    print(reward)

    # Results of sampling form the memory
    results = memories[0].sample(10, 1e-2)
    print(list(zip(*results))[2])

    return

def oneStep():
    
    print('Lets take one step ...')
    with utils.Env(showEnv=True, trainMode=False) as env:
        env.reset()
        result1, result2 = env.step(utils.randomPolicy)
        headers = 'states, actions, rewards, next_states, dones'.split(', ')
        for h, r in zip(headers, result1):
            try:
                shape = '[{}]'.format(r.shape)
            except:
                shape = '[No shape]'
            print(f'{h} --> {shape} {r}')

    return

def oneGame():

    print('Lets play a game:')        
    with utils.Env(showEnv=True, trainMode=False) as env:
        env.reset()
        allResults = env.episode( utils.randomPolicy, 100 )

        # Results are for a set of independent users ...
        result1, result2 = allResults
        states, actions, rewards, next_states, dones = zip(*result1)

        print(rewards)

    return

def testActorCritic():

    print('Testing the actor and the critic ...')
    print('------------------------------------')
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    memories = [memory.ReplayBuffer(200) for _ in range(2)]

    actor = NN.Actor(**config['Actor'])
    critic = NN.Critic(**config['Critic'])

    with utils.Env(showEnv=False, trainMode=True) as env:
        
        # Generate memories ...
        for _ in tqdm(range(10)):
            env.reset()
            allResults = env.episode( utils.randomPolicy, 200 )

            for i, result in enumerate(allResults):
                    memories[i].delNVals( 2 )
                    memories[i].appendMany(result)

        # Now, sample form the memory
        states, actions, rewards, next_states, dones = zip(*memories[0].sample( 20 ))
        states = np.array(states)

        print('shape of the states: '.format(states.shape))
        states = torch.from_numpy(states).float().to(device)
        action = actor(states)

        print('Action : {}'.format(action))
        qValue = critic(states, action)
        print('Q value: {}'.format(qValue))

        del states, action, qValue


    return

def allTests():
    
    print('Doing some random tests here')

    if config['TODO']['someTest']['fillReplayBuffer']:
        fillReplayBuffer()

    
    if config['TODO']['someTest']['oneStep']:
        oneStep()

    if config['TODO']['someTest']['oneGame']:
        oneGame()

    if config['TODO']['someTest']['testActorCritic']:
        testActorCritic()



    return
