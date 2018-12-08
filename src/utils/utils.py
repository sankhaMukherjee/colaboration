from unityagents import UnityEnvironment
import numpy as np
import json

from collections import deque, namedtuple
from tqdm import tqdm

def randomPolicy(states):

    actions = np.random.randn(2, 2) 
    actions = np.clip(actions, -1, 1)                  

    return actions

def playEpisodes(showEnv=False, trainMode=True, policy=None, numEpisodes=5, averaging=2, verbose=False):

    allScores      = []
    averagedScores = []
    runningAverage = deque([], maxlen=averaging)

    if policy is None:
        policy = randomPolicy

    config = json.load(open('config.json'))
    env    = UnityEnvironment(
        file_name = config['UnityEnv']['file_name'], 
        no_graphics = not showEnv)

    # get the default brain
    brain_name = env.brain_names[0]
    brain      = env.brains[brain_name]
    env_info   = env.reset(train_mode=trainMode)[brain_name]

    num_agents  = len(env_info.agents)
    action_size = brain.vector_action_space_size

    for i in tqdm(range(numEpisodes)):                             # play game for 5 episodes
        env_info = env.reset(train_mode=trainMode)[brain_name]     # reset the environment    
        states  = env_info.vector_observations                  # get the current state (for each agent)

        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            actions     = policy(states)
            env_info    = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations            # get next state (for each agent)
            rewards     = env_info.rewards                        # get reward (for each agent)
            dones       = env_info.local_done                     # see if episode finished
            scores     += env_info.rewards                        # update the score (for each agent)
            states      = next_states                             # roll over states to next time step

            if np.any(dones):                                  # exit loop if episode finished
                break

        allScores.append(scores)
        runningAverage.append(scores)
        averagedScores.append( np.mean( runningAverage ) )
        allScores.append(scores)

        if verbose:
            print('Total score (averaged over agents) this episode: {} -> {} '.format( 
                np.mean(scores), np.mean(runningAverage) ))

    env.close()


    return allScores, averagedScores

class Env:

    def __init__(self, showEnv=False, trainMode=True):
        self.no_graphics = not showEnv
        self.trainMode   = trainMode
        self.states      = None
        return

    def __enter__(self):

        config      = json.load(open('config.json'))
        self.env    = UnityEnvironment(
            file_name = config['UnityEnv']['file_name'], 
            no_graphics = self.no_graphics )

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain      = self.env.brains[self.brain_name]
        self.env_info   = self.env.reset(train_mode = self.trainMode)[self.brain_name]

        self.num_agents  = len(self.env_info.agents)
        self.action_size = self.brain.vector_action_space_size

        return self

    def reset(self):
        self.env.reset(train_mode=self.trainMode)
        self.states = self.env_info.vector_observations
        return self.states

    def step(self, policy):

        states      = self.states.copy()
        actions     = policy(states)
        env_info    = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations 
        rewards     = env_info.rewards             
        dones       = env_info.local_done          

        self.states = next_states

        results = []
        for i in range(self.num_agents):

            state       = states[i]
            action      = actions[i]
            reward      = rewards[i]
            next_state  = next_states[i]
            done        = dones[i]

            results.append((state, action, reward, next_state, done))

        return results

    def episode(self, policy, maxSteps=None):

        self.reset()
        stepCount     = 0
        allResults    = [[] for _ in range(self.num_agents)]

        while True:
            stepCount += 1
            results = self.step(policy)

            finished = False

            for agent in range(self.num_agents):
                state, action, reward, next_state, done = results[agent]
                allResults[agent].append(results[agent])
                finished = finished or done

            if finished or (stepCount >= maxSteps):
                break

        return allResults
            

    def __exit__(self, *args):
        self.env.close()
        return
