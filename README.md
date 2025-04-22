# AI-P4

Features

- Q-learning agent for autonomous navigation
- API client for world operations (locate, enter, move)
- Automatic game runner with episode tracking
- Score persistence between runs

## What It Does
This program is an automated player for a grid-based game that uses Q-learning to find optimal paths through a world. 

The client:

Connects to a game server using REST API endpoints ..
Navigates through a grid world using cardinal directions (N, S, E, W)
Learns from rewards received after each move
Optimizes its strategy over multiple episodes
Tracks and reports the best score achieved

## How It Works
The code consists of two main components:

API Client: Handles all communication with the server, including:

Getting the current location
Entering a world
Making moves
Retrieving scores and run history
Storing point data locally


Q-Learning Agent: Implements reinforcement learning to:

Maintain a Q-table mapping state-action pairs to expected rewards
Balance exploration vs. exploitation using epsilon-greedy strategy
Update Q-values based on rewards and future state values
Decrease exploration rate over time as it learns

Program Logic Flow

Initialization:

Set up API client with endpoints and credentials
Get current status (location, active run)
Enter the specified world if not already in one


Training Loop:

For each episode:

Reset position by re-entering the world
While not done:

Choose a direction (randomly or based on Q-values)
Make a move and receive reward
Update Q-values using the Q-learning formula
Track total reward for the episode


Reduce exploration rate (epsilon)
Track best reward achieved

Completion:

Store best score in local JSON file
Report final score from server



The Q-learning algorithm updates values according to:
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
Where:

α (alpha): Learning rate ..
γ (gamma): Discount factor for future rewards ..
ε (epsilon): Exploration rate that decays over time








