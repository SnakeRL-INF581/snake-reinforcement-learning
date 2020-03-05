import torch
from trainers.sarsa_trainer import SarsaTrainer
from trainers.q_learning_trainer import QLearningTrainer
from trainers.dql_trainer import DQLTrainer
from trainers.q_learning_state_epsilon_trainer import QLearningStateEpsilonTrainer
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import export_q_table, import_q_table

# Reinforcement learning parameters
epsilon_init = 0.2
epsilon_decay = 0.9
tau_init = 10
tau_decay = 0.9

alpha_init = 0.15
alpha_decay = 0.95
gamma = 0.9
decay_rate = 100

# Game parameters
size_x = 40
size_y = 25

learning_rate = 0.00005

# import pretrained q_dict
# q_dict_imported = import_q_table("pretrained_models/q_table.json")

trainer = SarsaTrainer(epsilon_init, epsilon_decay, alpha_init, alpha_decay,
    gamma, decay_rate, size_x, size_y, q_dict=None)

# trainer = QLearningTrainer(tau_init, tau_decay, alpha_init, alpha_decay,
#  gamma, decay_rate, size_x, size_y)

# trainer = QLearningStateEpsilonTrainer(epsilon_init, epsilon_decay, alpha_init, alpha_decay,
#      gamma, decay_rate, size_x, size_y)

# trainer = DQLTrainer(epsilon_init, epsilon_decay, learning_rate,
#    gamma, decay_rate, size_x, size_y, load_model_path="pretrained_models/trained_weights.pth")

# Main logic
iter = 0
start_time = time.monotonic()
visual_rate = 2500
try:
    while True:
        if (iter % visual_rate >= visual_rate - 1):
            trainer.iterate(visual=True, speed=25)
        trainer.iterate(visual=False)
        iter += 1

except KeyboardInterrupt:
    if not trainer.use_deep_learning:
        print("[!] Received interruption signal. Time elapsed : %ds" %
                (time.monotonic() - start_time))

        # Plot results
        t = np.arange(0, iter, trainer.decay_rate)

        fig, ax = plt.subplots()
        ax.plot(t[:-1], trainer.avg_scores)

        ax.set(xlabel='Number of episodes',
                ylabel='Average score over the last {} iterations'.format(
                    trainer.decay_rate),
                title='Evolution of average score across episodes')
        ax.grid()

        fig.savefig("test.png")
        export_q_table(trainer.q_dict, "q_table")
        plt.show()
        # plt.close()
    else:
        torch.save(trainer.policy_net.state_dict(), "weights.pth")
