# Simulate restaurant waiting times

![alt text](https://images.firstwefeast.com/complex/image/upload/c_limit,f_auto,fl_lossy,q_auto,w_1100/cmslqxf3wpgifoakb2qq.jpg)

A project for simulating a queueing system to find the optimal number of cooks to reduce average wait time at a restaurant.

## Getting Started

- Create a python virtual environment in the system and activate it.

**Installation using pip:**
  - `pip install virtualenv`
  - `virtualenv <env_name>`
  - `source <env_name>/bin/activate`

Install the dependencies for the project using the requirements.txt
  - `pip install -r requirements.txt`


### Run simulation
Run python restaurant_problem/main.py

There are several optional parameters
- -s: "simulation_samples"  The number of simulations to run per number of cooks. Default is 50.
- -p: performance_decay_rate: Rate at which the cooks performance degrades as more cooks work in the kitchen. Default decay rate is .03
- -d: debug: Whether to run process on debug mode. Debug mode will save queueing info to logs. Limits simulation samples to 2. 

RESTAURANT_PARAMETERS including current menu list, cook time, and expected number of orders per hour are defined in 
constants.py file

