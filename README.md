[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

## Introduction

This project trains agents to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. To give you an idea
of what the environment looks like a video of a 10 trained agent is below. The rotating green orbs are the goal states, and the agents are the simulated robot arms trying to continuously stay in the goal states. 
This project originally comes from the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

<p align="center">
    <img src = "https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif">
</p>

**Reward:** of +0.1 is provided for each step that the agent's hand (blue orb) is in the goal location. The goal of your agent is to maintain its position at the rotating target location for as many time steps as possible.

**Observation Space:** consists of 33 continuous variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

**Action Space:** is a vector with four continuous numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

**Goal:** The task is episodic. In order to solve the environment your agent (or agents) must get an average score of +30 over 100 consecutive episodes.


### Distributed Training

This project has two separate versions of the Unity environment:
- The first version contains a single agent (and is much slower to train).
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

**Solving the Second Version:**

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). 
Specifically, after each episode we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
This yields an **average score** for each episode (where the average is over all 20 agents). The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


----------------

### Getting Started

1. Download all the dependencies.
    * [OpenAI gym.](https://github.com/openai/gym) Install instructions in the repository README.
    * [Udacity Deep RL Repo.](https://github.com/udacity/deep-reinforcement-learning#dependencies) Install instructions in dependencies section of README. Details for just setting up this repo below.
        ```bash
        git clone https://github.com/udacity/deep-reinforcement-learning.git
        cd deep-reinforcement-learning/python
        pip install .
        ```

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

3. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. Alternatively you can create your own project directory and place the files in there.

--------------

### Instructions

To train an agent follow along in `DDPG_Reacher.ipynb`. You can watch an untrained agent, then train/see results of training an agent, and finally watch the trained agent act in the environment. If
you just want to see a summary of the project check out `report.md`.

#### File Descriptions

 - [DDPG_Reacher.ipynb](https://github.com/JSheldon3488/DeepRL_Continuous_Control/blob/master/DDPG_Reacher.ipynb) This is the notebook for viewing the project in action.
 - [agents.py](https://github.com/JSheldon3488/DeepRL_Continuous_Control/blob/master/agents.py) This file contains the class for creating an agent that will be able to 
 act and learn in the environment.
 - [networks.py](https://github.com/JSheldon3488/DeepRL_Continuous_Control/blob/master/networks.py) This file contains the actor and critic network classes.
 - [utils.py](https://github.com/JSheldon3488/DeepRL_Continuous_Control/blob/master/utils.py) This file contains the utility classes needed for this projet which include
 the replay buffer and the class to generate noise.
 - [report.md](https://github.com/JSheldon3488/DeepRL_Continuous_Control/blob/master/report.md) This file contains a description of the project, the results of the project, and a
 section of futue ideas for the project.