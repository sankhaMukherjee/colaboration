from unityagents import UnityEnvironment
import numpy as np
import json

from collections import deque, namedtuple
from tqdm import tqdm

def randomPolicy(states):

    actions = np.random.randn(2, 2) 
    actions = np.clip(actions, -1, 1)                  

    return actions

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

    def episode(self, policy, maxSteps=None, policy1=None, before=15):

        self.reset()
        stepCount     = 0
        allResults    = [[] for _ in range(self.num_agents)]

        # print('-----')
        while True:
            stepCount += 1
            
            if (policy1 is not None) and (stepCount < before):
                results = self.step(policy1)
            else:
                results = self.step(policy)

            finished = False

            for i, agent in enumerate(range(self.num_agents)):
                state, action, reward, next_state, done = results[agent]
                allResults[agent].append(results[agent])
                finished = finished or done

                reward = np.array(reward)
                # if np.any(reward>0.09):
                #     print('step: {:04d} | agent: {} | reward {}'.format(stepCount, i, reward))

            if finished or (stepCount >= maxSteps):
                break

        return allResults
            

    def __exit__(self, *args):
        self.env.close()
        return
