import json
import numpy as np

from unityagents import UnityEnvironment

from utils import utils as utils

def main():

    config = json.load(open('config.json'))

    if config['TODO']['train']:
        print('abcd')


    if config['TODO']['test']:
        # You should insert the policy network here ...
        testConfig = config['test']
        allScores, averagedScores = utils.playEpisodes(**testConfig)
        print(allScores)
        print(averagedScores)

    if config['TODO']['someTest']:

        from utils import memory
        from utils import utils
        print('Doing some random tests here')
        # epMemory = memory.Episode(maxLen=10, nSamples=2, nSteps=1, gamma=1)

        if False:
            print('Lets take one step ...')
            with utils.Env(showEnv=True, trainMode=False) as env:
                env.reset()
                result1, result2 = env.step(utils.randomPolicy)
                headers = 'states, actions, rewards, next_states, dones'.split(', ')
                for h, r in zip(headers, result1):
                    print(f'{h} --> {r}')

        if False:
            print('Lets play a game:')        
            with utils.Env(showEnv=True, trainMode=False) as env:
                env.reset()
                allResults = env.episode( utils.randomPolicy, 100 )

                # Results are for a set of independent users ...
                result1, result2 = allResults
                states, actions, rewards, next_states, dones = zip(*result1)
                print(dones)
                print(rewards)


        #     env.reset()
            
        #     allResults = env.episode( utils.randomPolicy, maxSteps=20 )
        #     states, actions, rewards, next_states, dones = zip(*allResults)
        #     print(dones)
        #     for a in actions:
        #         print(a)



    return

if __name__ == '__main__':
    main()

