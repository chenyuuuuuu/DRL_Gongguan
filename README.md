# DRL_Gongguan: Deep Reinforcement Learning for Gongguan Roundabout Traffic Control

This repository contains the code, configurations, and experiment results for **deep reinforcement learning (DRL)** applied to **traffic signal control at the Gongguan Roundabout** using **SUMO** and **Stable-Baselines3**.

The objective of this project is to develop and evaluate DRL-based controllers, including **Recurrent PPO (RPPO)** and **PPO**, to reduce queue lengths, improve average speed, and minimize travel time under realistic traffic demand scenarios.

---

## 📍 Gongguan Roundabout and SUMO Environment

Below is the real-world **Gongguan Roundabout** in Taipei, Taiwan, and its corresponding SUMO simulation environment used for DRL training and evaluation:

### Real Gongguan Roundabout
![Roundabout](https://github.com/chenyuuuuuu/DRL_Gongguan/blob/main/figures/Roundabout.png?raw=true)

### SUMO Simulation View
![Sumo_roundabout](https://github.com/chenyuuuuuu/DRL_Gongguan/blob/main/figures/Sumo_roundabout.png?raw=true)

---

## 🚀 Training Results: RPPO vs PPO

The following plots show the **training curves of RPPO and PPO under the Gongguan Roundabout scenario**.

> **In all plots below: the orange line represents RPPO, and the blue line represents PPO.**

### Average Queue Length
![avg_queue_length](https://github.com/chenyuuuuuu/DRL_Gongguan/blob/main/figures/avg_queue_length.png?raw=true)

### Average Speed
![avg_speed](https://github.com/chenyuuuuuu/DRL_Gongguan/blob/main/figures/avg_speed.png?raw=true)

### Average Step Reward
![avg_step_reward](https://github.com/chenyuuuuuu/DRL_Gongguan/blob/main/figures/avg_step_reward.png?raw=true)

### Average Travel Time
![avg_travel_time](https://github.com/chenyuuuuuu/DRL_Gongguan/blob/main/figures/avg_travel_time.png?raw=true)

These metrics collectively demonstrate the learning progression and performance differences between RPPO and PPO during training in the roundabout traffic control task.

---

## 🛠️ Project Structure

```plaintext
cfg/                # SUMO configuration files
demand/             # Traffic demand definitions
net/                # Network files
script/             # Training and evaluation scripts
roundabout_env/     # Custom SUMO Gym environment
figures/            # Result figures and screenshots
requirements.txt    # Python dependency list
.gitignore          # Git ignore rules


