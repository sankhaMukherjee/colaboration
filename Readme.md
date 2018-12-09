# colaboration

This project explores the possibility of using two Agents that use deep reinforcement learning to leanr to collaborate to play tennis. 

## 1. Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

Since this work was initially done on a Mac, the `./p3_collab-compet` folder contains a binary for Mac:

 - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
 - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
 - [Win32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
 - [Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Install these in a convinient location within your folder structure, and use this file for the training etc. Modify the Unity environment to point to this file.

## 2. Installing

1. Clone this repository to your computer, and create a virtual environment. Note that there is a `Makefile` at the base folder. You will want to run the command `make env` to generate the environment. Do this at the first time you are cloning the repo. The python environment is already copied from the deep reinforced learning repository, and is installed within your environment, when you generate the environment.

2. The next thing that you want to do is to activate your virtual environment. For most cases, it will look like the following: `source env/bin/activate`. If you are using the [`fish`](https://fishshell.com) shell, you will use the command `source env/bin/activate.fish`.

3. Change to the folder `src` for all other operations. There is a `Malefile` available within this folder as well. You can use the Makefile as is, or simply run the python files yourself. Availble optins in the `Makefile` are as follows:

 - `make run`   : run the program for training
 - `make clean` : delete all temorary files (including the results. Use this with care ...)

## 3. Operation

After you have generated the environment, activated it, and switched to the `src` folder, you will want to execute the program. The folder structure is shown below:

```bash
├── Makefile  # <------ allows you to run the program and perform cleanup
├── colaborate.py # <-- The main file that stitches everything together
├── config.json # <---- All configuration parameters go here ...
├── tests # <---------- Code for temporary testing is present here
│   └── allTests.py
└── utils # <---------- Various utilities for running the program
    ├── NN.py # <------ The Actor, Critic and the Agent is defined here 
    ├── memory.py # <-- The Replay buffer is present here 
    ├── tester.py # <-- This contains functions that allow a saved state to be taken and played
    ├── trainer.py # <- This contains functions that allow the training to take place smoothly
    └── utils.py # <--- Various utility programs are available here
```

For successfully running the program, all you really need to do is to change different parameters within the configuration file  `src/config.json`. The configuration file is conviniently segmented into various parts. Most of the parameters are easy to understand. The major sections are as follows:

 - `TODO`: Allows you to either train or test your programs 
 - `UnityEnv` : locatio of the unity environment file 
 - `training` : Parameters specific to the training process 
 - `testing` : Parameters specific to the testing process 
 - `Agent` : Parameters specific to the agent. Rememebr that in this game, there will be two of these. These parameters also include parameters for the `Actor`, the `Critic` and the `ReplayBuffer`
 - `test` : Unimportant and will be removed later.

### 3.1. Training the model

Training the model is acomplished by setting the `config['TODO']['train']` parameter to `true`. Note that you will also have to specify the specifications of the `Agents`, that will be trained. Once these parameters have been properly set, training can begin. You are welcome to change the parameters and see how the training proceeds. 

Once training has completed, the model, along with the scores and a picture of the scores as a function of training epoch is saved in the file `results\`. Every time, a new folder is created using the current date-time, and all information for recreating the model is saved. 

## 4. Model Description

The model learns throuhg the action of two independent agents. 

### 4.1. The `ReplayBuffer`

This maintains a `deque` that continuously adds new experiences (i.e. a tuple containing the current state, the current action, the next state, the next action, the next reward, and whether we are done with the current episode). It has two methods. The first allows one to add experiences from this deque, and another that allows it to sample from it. 

### 4.2. The `Actor`

This is a simple 3-layer fully connected network. The input is the current state, and the output is a vector of the same size as the action space, and represents a Q value for each action. The first two layers have relu activation, while the last one is unactivated, which allows the last layer to have any real value. 

### 4.3. The `Critic`

This is a simple 3-layer fully connected network. The input is the current state as well as the current action, and the output is a vector of the Q value for the state-action pair. The first two layers have relu activation, while the last one is unactivated, which allows the last layer to have any real value. 

### 4.3. The `Agent`

The agent comprises of two actors and two critics. There is a slow-moving and fast-moving components of the actors and the critics. The agent comprises of several methods that allow the program to handle several things successfully. 

A `step` method allows the agent to obtain `nSamples` form the replay buffer and train the fast-varying actor and critic. This is followd by a softupdate, that allows the slow-moving components to be slowly updated with the weights of the fast-moving components. 

A `softUpdate` method allows the quick update of the weights of the slow moving components of the critic and the actor by using those of the fast-moving actors and the critics. 

An `updateBuffer` allows the `ReplayBuffer` to be updated with new data. At every epoch, a new episode is played, and the replay buffer updated.

### 4.4. The learning algorithm

The Learning algorithm is shown in the file `utils\trainer.py`. This defines a function `explorePolicy()` that takes a parameter `explore` and retuens a `policy` function that combines the result of the action of a slow-moving actor, along with a random action in proportion to explore. This allows us to generate data for the replay beffers that have a good combination of exploration vs. exploitation.

The algorithm roughly translates to the following:

1. Generate two instances of the Agent
2. Initialize the Unity Environment
3. Do the following for many iterations
    1. play for one episode with a particulat exploration factor
    2. Decrease the exploration factor slightly if necessary
    3. Split the results and save the data into two replay buffers of the two agents.
    4. for each agent, learn the actors and critics 
    5. Save the scores 
4. Save the scores and the models.

### 4.5. Examples of Learning

It turns out that exploration is essential for learning. Without exploration, the learning algorithms have absolutely no bearing on learning as shown below. 

![bad algorithm](https://raw.githubusercontent.com/sankhaMukherjee/colaboration/master/results/2018-12-09--18-50-45/scores.png)

When the exploreation is turned off too fast, the learning decreases at a significant rate as shown below:

![bad algo](https://raw.githubusercontent.com/sankhaMukherjee/colaboration/master/results/2018-12-09--21-08-29/scores.png)

For higher exploration, the learning is slightly better. 

![better](https://github.com/sankhaMukherjee/colaboration/blob/master/results/2018-12-09--21-32-29/scores.png)

However, unfortuantely, the result is not very good, and the system did not learn very well ...

More needs to be done to stabilize the system.

## 5. Future Work

1. I shall try to play with the different parameters, and also run the model for much longer to see whether the model magically learns something, as in the case of the course instructors. 
2. I shall also try to see whether it is possible to comvine the last few views to generate something meaningful as input, rather than just using the latest observation. This means that the agent will have knowledge of not only the current observation, but also the last few observations. This should allow the agent to learn a much better. 

## 6. Authors

Sankha S. Mukherjee - Initial work (2018)

## 7. License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## 8. Acknowledgments

 - This repo contains a copy of the python environment available [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/python). 
 - The solutions follow many of the solutions available in the UDacity course.

  

 