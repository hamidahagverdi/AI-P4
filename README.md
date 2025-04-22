# AI-P4

Features

- Q-learning agent for autonomous navigation
- API client for world operations (locate, enter, move)
- Automatic game runner with episode tracking
- Score persistence between runs

## What It Does
This program is an automated player for a grid-based game that uses Q-learning to find optimal paths through a world. 

The client:

Connects to a game server using REST API endpoints
Navigates through a grid world using cardinal directions (N, S, E, W)
Learns from rewards received after each move
Optimizes its strategy over multiple episodes
Tracks and reports the best score achieved
