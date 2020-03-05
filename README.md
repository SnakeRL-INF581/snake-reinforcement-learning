# Learn to play Snake using Reinforcement Learning

## Overview

The concept is as follows: the player maneuvers a line which grows in length,
with the line itself being a primary obstacle. The line resembles a moving
snake that becomes longer as it eats food. The player loses when the snake runs
into the screen border or itself. The goal is to get as long as possible before
dying. In this report, we introduce our different reinforcement learning
approaches in order to efficiently train an agent to play Snake.

## Quick Start
Install all required packages:
```sh
pip3 install -r requirements.txt
```

In `main.py` choose the hyper-parameters, the model and run:
```sh
python3 main.py
```
Snake is learning!

## Train Models

4 RL models are currently implemented in the framework. 3 classic models and 1 
deep learning model. All models inherit from the `ABCTrainer` class which 
contains all the abstract methods needed to create a new model (`choose_action`, 
`update_hyperparameters`...). Check in the trainer folder what the 
hyperparameters needed are for each models.

## Play With A Pretrained Model

- For classic models: in `main` import a pretrained `q_table` by using the 
`import_q_table` function to instanciate the trainer.
- For `DQLTrainer`: use the `load_model_path` attribute of `DQLTrainer`

```
DQLTrainer(epsilon_init, epsilon_decay, learning_rate, gamma, decay_rate, size_x, 
size_y, load_model_path="pretrained_models/trained_weights.pth")
```
To save you hours of training required before getting results, we provide 
`pytorch` pretrained weights that you can use.

It is also possible to stop the training of your model, by using the `test` 
argument of the trainers `iterate` method.

## Find the best hyper-parameters

`grid_search.py` allows you to test a wide range of hyper-parameters on the 
model of your choice, it returns a `json` file with the results.
