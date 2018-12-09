import numpy as np
import json, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import utils, memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        # self.reset_parameters()

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        # self.reset_parameters()

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1) # axis = 1
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent:

    def __init__(self):

        self.config = json.load(open('config.json'))['Agent']

        self.actorFast = Actor(**self.config['Actor'])
        self.actorSlow = Actor(**self.config['Actor'])
        self.actorOptimizer = optim.Adam( 
            self.actorFast.parameters(), lr=self.config['actorLR'] )
        
        self.criticFast = Critic(**self.config['Critic'])
        self.criticSlow = Critic(**self.config['Critic'])
        self.criticOptimizer = optim.Adam( 
            self.criticFast.parameters(), lr=self.config['criticLR'] )

        # Create some buffer for learning
        self.buffer = memory.ReplayBuffer(self.config['ReplayBuffer']['maxEpisodes'])

        return

    def step(self, nSamples, Tau=None):
        '''Take one step toward learning
        
        This takes a set of samples provided by the parameter 
        ``nSamples`` and learns the fast actor and critics. This
        will also update actorSlow and criticSlow at the end of
        the nSamples. Hence, nSamples is a good indication of how
        fast the slow-moving actors and critics should be updated
        in the learning process.
        
        Parameters
        ----------
        nSamples : {int}
            Number of samples that will be used for sampling the memory
            buffer, and the subsequently used for learning the actor and
            the critic.
        Tau : {float or None}, optional
            This is the parameter that will be used for the soft update of
            the slow-moivng components. (the default is None, wehrein, the 
            value from the config file will be used for the update)
        '''

        result = zip(*self.buffer.sample( nSamples ))
        states, actions, rewards, next_states, dones = map(np.array, result)
        
        states       = torch.from_numpy(states).float().to(device)
        actions      = torch.from_numpy(actions).float().to(device)
        rewards      = torch.from_numpy(rewards).float().to(device)
        next_states  = torch.from_numpy(next_states).float().to(device)

        # ------------ Update the critics --------------------------
        nextActionHat  = self.actorSlow( next_states )
        qValHat        = rewards + self.criticSlow( next_states, nextActionHat ) # * gamma (=1)
        qVal           = self.criticFast( states, actions )

        lossFnCritic = F.mse_loss(qValHat, qVal)
        self.criticOptimizer.zero_grad()
        lossFnCritic.backward()
        self.criticOptimizer.step()

        # ------------ Update the actors --------------------------
        actionHat   = self.actorFast( states )
        lossFnActor = - self.criticSlow( states, actionHat ).mean()

        self.actorOptimizer.zero_grad()
        lossFnActor.backward()
        self.actorOptimizer.step()

        del states, actions, rewards, next_states
        del nextActionHat, qValHat, qVal
        del actionHat
        del lossFnCritic, lossFnActor

        # ------------ Soft update the slow components -------------
        self.softUpdate()

        return

    def softUpdate(self, tau=None):

        if tau is None:
            tau = self.config['Tau']

        for v1, v2 in zip(self.actorFast.parameters(), self.actorSlow.parameters()):
            v2.data.copy_( tau*v1 + (1-tau)*v2 )

        for v1, v2 in zip(self.criticFast.parameters(), self.criticSlow.parameters()):
            v2.data.copy_( tau*v1 + (1-tau)*v2 )

    def updateBuffer(self, data, nReduce=0):
        self.buffer.appendMany( data )
        return

    def save(self, folder, name):

        torch.save( self.actorFast.state_dict(),  os.path.join( folder, f'{name}.actorFast')  )
        torch.save( self.actorSlow.state_dict(),  os.path.join( folder, f'{name}.actorSlow')  )
        torch.save( self.criticFast.state_dict(), os.path.join( folder, f'{name}.criticFast')  )
        torch.save( self.criticSlow.state_dict(), os.path.join( folder, f'{name}.criticSlow')  )

        return

    def load(self, folder, name):

        self.actorFast.load_state_dict(torch.load( os.path.join( folder, f'{name}.actorFast')  ))
        self.actorSlow.load_state_dict(torch.load( os.path.join( folder, f'{name}.actorSlow')  ))
        self.criticFast.load_state_dict(torch.load( os.path.join( folder, f'{name}.criticFast')  ))
        self.criticSlow.load_state_dict(torch.load( os.path.join( folder, f'{name}.criticSlow')  ))

        return

