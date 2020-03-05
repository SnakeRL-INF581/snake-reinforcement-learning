# Be careful! This script might run for a long time (~2 hours).

from trainers.q_learning_trainer import QLearningTrainer
from trainers.sarsa_trainer import SarsaTrainer
import json

# Reinforcement learning parameters
epsilon_inits = [0.2, 0.15]
epsilon_decays = [0.9]
alpha_inits = [0.15, 0.20]
alpha_decays = [0.95]
tau_inits = [10, 100, 1]
tau_decays = [0.9, 0.8]
gammas = [0.9]
decay_rate = 100

# Game parameters
size_x = 40
size_y = 25

# Training parameters
training_size = 8000
validation_size = 1000
nb_train = 3

scores_list = []

for tau_init in tau_inits:
    for tau_decay in tau_decays:
        for alpha_init in alpha_inits:
            for alpha_decay in alpha_decays:
                for gamma in gammas:
                    avg_val_best_score = 0
                    avg_val_avg_score = 0
                    for i in range(nb_train):
                        train = QLearningTrainer(tau_init, tau_decay, alpha_init,
                            alpha_decay, gamma, decay_rate, size_x, size_y)
                        for iter in range(training_size):
                            train.iterate(visual=False)
                        val_best_score = 0
                        val_avg_score = 0
                        for iter in range(validation_size):
                            score = train.iterate(visual=False, test=True)
                            if score > val_best_score:
                                val_best_score = score
                            val_avg_score += score

                        val_avg_score /= validation_size
                        avg_val_best_score += val_best_score
                        avg_val_avg_score += val_avg_score

                    avg_val_avg_score /= nb_train
                    avg_val_best_score /= nb_train

                    conf_dict = {
                        "tau_init": tau_init,
                        "tau_decay": tau_decay,
                        "alpha_init": alpha_init,
                        "alpha_decay": alpha_decay,
                        "gamma": gamma,
                        "avg_score": avg_val_avg_score,
                        "best_score": avg_val_best_score
                    }
                    scores_list.append(conf_dict)

file_name = "grid_test"

with open(file_name + ".json","w") as fp:
    json.dump(scores_list, fp, skipkeys=True)
print("json file exported!")
