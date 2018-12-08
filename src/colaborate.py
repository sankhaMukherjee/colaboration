import json
import numpy as np

from unityagents import UnityEnvironment

from utils import utils as utils
from tqdm import tqdm

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

        if True:

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
                states, actions, rewards, next_states, dones, cumReward = zip(*result1)

    return

if __name__ == '__main__':
    main()

