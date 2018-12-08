# continuousControl

This project is useful for training a deep Q network that learns to move to a particular position. 

## 1. Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

Since this work was initially done on a Mac, the `./p1_continuous-control` folder contains a binary for mac. This is the version that will be uses 20 different conditions. There are several environments available:

 - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
 - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
 - [Win32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
 - [Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Install these in a convinient location within your folder structure, and use this file for the training etc. Modify the Unity environment to point to this file.

## 2. Installing

1. Clone this repository to your computer, and create a virtual environment. Note that there is a `Makefile` at the base folder. You will want to run the command `make env` to generate the environment. Do this at the first time you are cloning the repo. The python environment is already copied from the deep reinforced learning repository, and is installed within your environment, when you generate the environment.

2. The next thing that you want to do is to activate your virtual environment. For most cases, it will look like the following: `source env/bin/activate`. If you are using the [`fish`](https://fishshell.com) shell, you will use the command `source env/bin/activate.fish`.

3. Change to the folder `src` for all other operations. There is a `Malefile` available within this folder as well. You can use the Makefile as is, or simply run the python files yourself.

## 3. Operation

After you have generated the environment, activated it, and switched to the `src` folder, you will want to execute the program. The folder structure is shown below:

```bash
.
├── Makefile  # <------------ Run tests and other things
├── allTests.py # <---------- Function for testing
├── config.json # <---------- All configuration options present here
├── continuousControl.py # <- Main file for the program
├── lib # <------------------ A set of library files
│   ├── NN.py # <------------    File for the actor, critic and the Agent
│   ├── memory.py # <--------    File for the replay buffer
│   └── utils.py # <---------    Other utilities
└── tests # <---------------- Some testing code
    ├── __init__.py
    └── memTests.py
```

You will typically run the program by the following command `make run`. Most of the essential parameters can be modified by changing the information within the `config.json` file. The configuration file has several sections: 

 - `tests`: Decide if you want to run the tests 
 - `Agent`: Configuration fort he agent, including the actors and the critics and the replay buffers
 - `learnParams`: Configurations for the optimizer
 - `mainLearn`: Configuration for the how sampling and replay buffer filling will take place

Several different types of operations that you can do is described below:

### 3.1. Training the model

For training the model, just type `make run` in the `src` folder. You are welcome to play with the different opetions to see how the different optins affect the result. An example of the model performance is shown below. 

![Learning](https://raw.githubusercontent.com/sankhaMukherjee/continuousControl/master/results/img/2018-12-07--00-32-1544113924.png)

The orange line is one that is the max Q value from an episode, while the blue line is the average of the Q value of 20 robotes. As can be seen, more experimentation is needed to make this work.

## 4. Model Description

A short description of the model will be provided here. The learning is done by an `Agent` (defined in `src/lib/NN.py`). This agent uses an `Actor` that acts as a policy network, and a `Critic` which acts as the q-learner. Each of these is further divided into two different networks, the fast learner (`actor1`, and `critic1`) that is updated every step, and a slow learner (`actor2`, and `critic2`), whose parameters are updated after every few strps. The agent employs a replay buffer `ReplayBuffer` (defined in `src/lib/memory.py`). This is able to sample episodes rather than just events. This will allow us to try to learn not just with the unbiased Monte Carlo estimation, but also with the an N-step estimate.

### 4.1. The `ReplayBuffer`

This maintains a `deque` that adds new episodes to itself. There is a dedicated method in `src/lib/utils.py` that will add a set of episodes to the buffer. The buffer is also able to purge episodes that dont perform very well. This will allow the replay buffer to retain the best episodes to train from, rather than losing them.

As noted before, the replay buffer saves entire episodes rather than just experiences. This allows the replay buffer during sampling to obtain the cumulative reward for the next few steps, which would not be possible if we were to just save experiences. Sampling is fairly sophisticated, and this buffer samples episodes that were able to provide the best learning experiences. 

### 4.2. The `Actor`

This is a three layer network that takes the state and returns a set of four numbers between -1 and 1, corresponding to the value associated with each action. As mentioned before, there are two of these - the fast moving and the slow moving components. All parameters can be controlled with the config file.

### 4.3. The `Critic`

This is a three layer network that takes the state and a corresponding action vector and returns a single Q value, corresponding to the value associated with each action. As mentioned before, there are two of these - the fast moving and the slow moving components. All parameters can be controlled with the config file.

### 4.3. The `Agent`

The agent is composed of two actors and two critics. Important methods within the `Agent` are:

 - A `learn` method allows one to learn by drawing a desired set of samples form the replay buffer.

 - An `updateBuffer` method calculates generates an episode and adds results to the replay buffer.

 - A `transferWeights` step updates weights from the slow-moving networks with thse form the fast-moving networks.


### 4.4. The learning algorithm

The algorithm may be briefly described as follows:

1. For each step:
    1.1. Update the replay buffer every N steps (including the 0th step)
    1.2. Draw N samples form the memory buffer 
        1.2.1. Update the critic network 
        1.2.2. Update the actor network 
    1.5. Update the slow-moving actor and critic networks

## 5. Future Work

The following might be added to improve the learning algorithm:

1. Improve the speed of the network 
2. Add some clipped-surrogate implementation to improve stability

## 6. Authors

Sankha S. Mukherjee - Initial work (2018)

## 7. License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## 8. Acknowledgments

 - This repo contains a copy of the python environment available [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/python). 
 - The solutions follow many of the solutions available in the UDacity course.

  

 