# AMR Project

## Project Objectives

The objective of this project is that you deploy some of the functionalities that were discussed during the course on a real robot platform. In particular, we want to have functionalities for path and motion planning, localisation, and environment exploration on the robot.

We will particularly use the Robile platform during the project; you are already familiar with this robot from the simulation you have been using throughout the semester as well as from the few practical lab sessions that we have had.


## Video Demo

[![Autonomous Mobile Robotics Project | Path Planning, Localization & Exploration on Robile](https://img.youtube.com/vi/cXD8E2jGFP0/hqdefault.jpg)](https://youtu.be/cXD8E2jGFP0)

## Usage Instructions

### Running in Simulation
To run the robot in simulation:
```bash
cd <your_ws>
export GAZEBO_MODEL_PATH=./src/robile_gazebo/models
ros2 launch robile_gazebo gazebo_4_wheel.launch.py
````

---

### Running on Real Robot

First, SSH into the Robile robot:

```bash
ssh studentkelo@<robile_ip>
```

Then, bring up the robot base:

```bash
cd ~/ros2ws/
source install/setup.sh
ros2 launch robile_bringup robot.launch.py
```

---

### Task 1 – Path and Motion Planning (AMCL)

1. Run the localization node:

   ```bash
   ros2 launch robile_localization amcl_localization.launch.py
   ```
2. In another terminal, run the navigation stack:

   ```bash
   ros2 launch robile_path_planner amr_navigation_launch.py localization_source:=amcl
   ```

---

### Task 2 – Custom Monte Carlo Localization (MCL)

Run the MCL localization node:

```bash
ros2 launch robile_localization mcl_localisation.launch.py
```

Don’t forget to provide the **initial pose** (via RViz or CLI).

---

### Task 3 – Environment Exploration (with SLAM)

1. Run SLAM to generate the map:

   ```bash
   ros2 run robile_localization slam.launch.py
   ```
2. In another terminal, run the navigation stack with SLAM localization:

   ```bash
   ros2 launch robile_path_planner amr_navigation_launch.py localization_source:=slam
   ```
3. In a third terminal, start the exploration node:

   ```bash
   ros2 run robile_explorer exploration_node.py
   ```


