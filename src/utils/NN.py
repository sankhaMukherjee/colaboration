import numpy as np
import json, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import utils, memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=24, fc2_units=48, fc3_units=48):
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
        self.bn1 = nn.BatchNorm1d(num_features=fc1_units)
        
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(num_features=fc2_units)

        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn3 = nn.BatchNorm1d(num_features=fc3_units)

        self.fc4 = nn.Linear(fc3_units, action_size)
        self.bn4 = nn.BatchNorm1d(num_features=action_size)
        # self.reset_parameters()

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.tanh(self.bn1(self.fc1(state)))
        x = F.tanh(self.bn2(self.fc2(x)))
        x = F.tanh(self.bn3(self.fc3(x)))
        return F.tanh(self.bn4(self.fc4(x)))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=24, fc2_units=48, fc3_units=48):
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
        self.bn1 = nn.BatchNorm1d(num_features=fcs1_units)

        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.bn2 = nn.BatchNorm1d(num_features=fc2_units)

        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn3 = nn.BatchNorm1d(num_features=fc3_units)

        self.fc4 = nn.Linear(fc3_units, 1)
        # self.reset_parameters()

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.tanh(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1) # axis = 1
        x = F.tanh(self.bn2(self.fc2(x)))
        x = F.tanh(self.bn3(self.fc3(x)))
        return self.fc4(x)

class Agent:

    def __init__(self):
        '''This is the Agent class. 
        
        This comprises of two actors and two critics. The actors are 
        '''

        self.config = json.load(open('config.json'))['Agent']

        self.actorFast = Actor(**self.config['Actor'])
        self.actorSlow = Actor(**self.config['Actor'])
        self.actorOptimizer = optim.Adam( 
            self.actorFast.parameters(), lr=self.config['actorLR'] )
        
        self.criticFast = Critic(**self.config['Critic'])
        self.criticSlow = Critic(**self.config['Critic'])
        self.criticOptimizer = optim.Adam( 
            self.criticFast.parameters(), lr=self.config['criticLR'] )
        
        if torch.cuda.is_available():
            self.actorFast = self.actorFast.cuda()
            self.actorSlow = self.actorSlow.cuda()
            self.criticFast = self.criticFast.cuda()
            self.criticSlow = self.criticSlow.cuda()

        # Create some buffer for learning
        self.buffer = memory.ReplayBuffer(**self.config['ReplayBuffer'])

        # We shall put everything in .train() mode only during training ...
        self.actorFast.eval()
        self.actorSlow.eval()
        self.criticFast.eval()
        self.criticSlow.eval()

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

        self.actorFast.train()
        self.actorSlow.train()
        self.criticFast.train()
        self.criticSlow.train()

        result = zip(*self.buffer.sample( nSamples ))
        states, actions, rewards, next_states, dones, cumRewards, totalHits = map(np.array, result)

        
        states       = torch.from_numpy(states).float().to(device)
        actions      = torch.from_numpy(actions).float().to(device)
        rewards      = torch.from_numpy(rewards).float().to(device)
        next_states  = torch.from_numpy(next_states).float().to(device)
        cumRewards   = torch.from_numpy(cumRewards.reshape(-1, 1)).float().to(device)

        # ------------ Update the critics --------------------------
        nextActionHat  = self.actorSlow( next_states )
        qVal           = rewards + self.criticSlow( next_states, nextActionHat ) # * gamma (=1)
        qValHat        = self.criticFast( states, actions )

        lossFnCritic = F.mse_loss(qValHat, qVal)
        # lossFnCritic = F.mse_loss(qValHat, cumRewards)
        self.criticOptimizer.zero_grad()
        lossFnCritic.backward()

        # Gradient clipping ...
        # nn.utils.clip_grad_norm(self.criticFast.parameters(), 0.25)
        # for p in self.criticFast.parameters():
        #     p.data.add_(-1e-4, p.grad.data)

        self.criticOptimizer.step()

        qLoss = F.mse_loss(qValHat, qVal).cpu().data.numpy()

        # ------------ Update the actors --------------------------
        actionHat   = self.actorFast( states )
        lossFnActor = - self.criticSlow( states, actionHat ).mean()

        self.actorOptimizer.zero_grad()
        lossFnActor.backward()

        # Gradient clipping ... 
        # This is very erratic ....
        # nn.utils.clip_grad_norm(self.actorFast.parameters(), 0.25)
        # for p in self.actorFast.parameters():
        #     p.data.add_(-1e-4, p.grad.data)
        self.actorOptimizer.step()

        aLoss = self.criticSlow( states, actionHat ).mean().cpu().data.numpy()

        del states, actions, rewards, next_states, cumRewards
        del nextActionHat, qValHat, qVal
        del actionHat
        del lossFnCritic, lossFnActor

        self.actorFast.eval()
        self.actorSlow.eval()
        self.criticFast.eval()
        self.criticSlow.eval()

        return qLoss, -aLoss

    def softUpdate(self, tau=None):
        '''update the slow actors with the fast actors
        
        This function updates the weights of the slow-moving actor and critic with
        a part of the weight of the fast-moving critic. The factor ``Tau`` is used
        for the ratio of the fast-moving weights to transfer.

        
        Parameters
        ----------
        tau : {float}, optional
            This is the ratio of the fast-moving actor and and critic that will be 
            incorporated into the slow-moving components. (the default is None, 
            which results in the value being taken form the config file.)
        '''

        if tau is None:
            tau = self.config['Tau']

        for v1, v2 in zip(self.actorFast.parameters(), self.actorSlow.parameters()):
            v2.data.copy_( tau*v1 + (1-tau)*v2 )

        for v1, v2 in zip(self.criticFast.parameters(), self.criticSlow.parameters()):
            v2.data.copy_( tau*v1 + (1-tau)*v2 )

    def updateBuffer(self, data, nReduce=0):
        '''update the memory buffer with new data
        
        This function takes a list of data values, and updates the memory buffer
        of the current agent. 
        
        Parameters
        ----------
        data : {list of tuples}
            Each value of the tuple contains the following: 
            ``(states, actions, rewards, next_states, dones, cumRewards)``. At one
            go, this function will insert all the values that have been associated
            with the current data. 
        nReduce : {number}, optional
            Depricated. Currently not used. This will be removed from future versions. 
            (the default is 0, which [default_description])
        '''
        self.buffer.appendMany( data )
        return

    def save(self, folder, name):
        '''Save the model
        
        This function allows one to save a model, including the folder, and the weights
        of the actors, critics as well as the memory buffer associated with the agent. A
        name is supplied because in this case, there is more than a single agent. Hence,
        the name supplied would be the name of the Agent.
        
        Parameters
        ----------
        folder : {str}
            The path where the model is to be saved.
        name : {str}
            The name of the agent that is being saved.
        '''

        torch.save( self.actorFast.state_dict(),  os.path.join( folder, f'{name}.actorFast')  )
        torch.save( self.actorSlow.state_dict(),  os.path.join( folder, f'{name}.actorSlow')  )
        torch.save( self.criticFast.state_dict(), os.path.join( folder, f'{name}.criticFast')  )
        torch.save( self.criticSlow.state_dict(), os.path.join( folder, f'{name}.criticSlow')  )

        self.buffer.save(folder, name)

        return

    def load(self, folder, name, map_location=None):
        '''Load an agent
        
        An agent saved with the save command can be loaded with the load command. This is useful
        because we may want to either hot-start a training from a previous model, or we might
        want to directly load the agent and test it. This will allow us to do that.
        
        Parameters
        ----------
        folder : {str}
            The path from where the model is to be loaded.
        name : {str}
            The name of the agent that is being loaded.
        '''

        if map_location is None:
            self.actorFast.load_state_dict(torch.load( os.path.join( folder, f'{name}.actorFast')    ))
            self.actorSlow.load_state_dict(torch.load( os.path.join( folder, f'{name}.actorSlow')    ))
            self.criticFast.load_state_dict(torch.load( os.path.join( folder, f'{name}.criticFast')  ))
            self.criticSlow.load_state_dict(torch.load( os.path.join( folder, f'{name}.criticSlow')  ))
        else:
            print('----------------')
            self.actorFast.load_state_dict(torch.load( os.path.join( folder, f'{name}.actorFast')   , map_location = map_location ))
            self.actorSlow.load_state_dict(torch.load( os.path.join( folder, f'{name}.actorSlow')   , map_location = map_location ))
            self.criticFast.load_state_dict(torch.load( os.path.join( folder, f'{name}.criticFast') , map_location = map_location ))
            self.criticSlow.load_state_dict(torch.load( os.path.join( folder, f'{name}.criticSlow') , map_location = map_location ))

        self.buffer.load(folder, name)

        return

