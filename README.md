# CDIO-2025

Code for Group 24 for DTU's CDIO 2025 project.

## Overview

This repository contains the codebase for an autonomous robot system developed as part of the CDIO 2025 course at DTU. The system is designed to perceive its environment, plan paths, and control a robot to complete various tasks on a course.

## Features

- **Computer Vision**: Detects objects, balls, goals, and obstacles using camera input.
- **Path Planning**: Uses A* and optimized strategies to find safe paths, accounting for robot size and obstacles.
- **State Machine**: Modular state-based control for robot behaviors (e.g., searching, collecting, delivering).
- **Pure Pursuit Navigation**: Smooth path following for the robot.
- **Obstacle Avoidance**: Dynamic inflation of obstacles to ensure safe navigation.
- **Logging**: Detailed logging for debugging and analysis.

## Folder Structure

```
ServerClient/
  client/
    client.py                # Client code for communication
  server/
    AImodel.py               # AI and perception models
    config.py                # Configuration parameters
    Course.py                # Course and object representation
    PathPlanner.py           # Path planning algorithms
    PurePursuit.py           # Path following logic
    server.py                # Main server and robot control loop
    states/                  # State machine implementations
  rules.txt                  # Project rules and documentation
README.md
```

## Getting Started

### Prerequisites

- Python 3.11 or newer
- OpenCV (`opencv-python`)
- NumPy
- Any other dependencies listed in `requirements.txt` (if present)

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/your-org/CDIO-2025.git
    cd CDIO-2025/ServerClient
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Running the Server

Start the main server (robot control and vision):

```sh
python server/server.py
```

### Running the Client

Start the client (for communication or visualization):

```sh
python client/client.py
```

## Configuration

Edit `server/config.py` to adjust parameters such as robot radius, thresholds, and logging levels.

## Development

- **States**: Add or modify robot behaviors in `server/states/`.
- **Path Planning**: Improve or swap algorithms in `server/PathPlanner.py`.
- **Vision**: Update perception logic in `server/AImodel.py` and `server/Course.py`.

## Logging

Logs are written to `robot.log` and include detailed information about state transitions, path planning, and errors.