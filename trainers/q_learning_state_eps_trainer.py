from utils.direction import Direction
import random
from trainers.abctrainer import ABCTrainer
import numpy as np

class QLearningStateEpsTrainer(ABCTrainer):

    def __init__(self, epsilon_init, epsilon_decay, alpha_init, alpha_decay,
                    gamma, decay_rate, size_x, size_y):

        super().__init__(gamma, decay_rate, size_x, size_y, False)

        # Training data
        self.q_dict = {}
        self.eps_dict = {}
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha_init
        self.alpha_decay = alpha_decay

    # Implement abstract method
    def update_hyperparameters(self):
        super().update_hyperparameters()
        self.alpha *= self.alpha_decay

    # Implement abstract method
    def get_state(self):
        super().get_state()
        return self._hash_bin(self.get_danger_vect() + self.get_food_vect())

    # Implement abstract method
    def choose_action(self, state):
        super().choose_action(state)
        try:
            eps = self.eps_dict[state]
        except KeyError:
            eps = self.epsilon_init
        sample = (np.random.uniform() < eps)
        self.eps_dict[state] = eps*self.epsilon_decay
        if sample:
            return np.random.choice(list(Direction))
        else:
            try:
                return max(self.q_dict[state], key=self.q_dict[state].get)
            except KeyError:
                self.q_dict[state] = dict.fromkeys(list(Direction),0)
                return np.random.choice(list(Direction))

    # Implement abstract method
    def update_q_dict(self, state, state2, action, action2):
        super().update_q_dict(state, state2, action, action2)
        if state not in self.q_dict:
            self.q_dict[state] = dict.fromkeys(list(Direction), 0)
        if state2 not in self.q_dict:
            self.q_dict[state2] = dict.fromkeys(list(Direction), 0)

        #Q-learning update
        prev = self.q_dict[state][action]
        self.q_dict[state][action] = prev + self.alpha * (
            self.reward +
            self.gamma * self.q_dict[state2][max(self.q_dict[state2],
                key=self.q_dict[state2].get)] - prev
        )

    # Implement abstract method
    def print_recap(self):
        super().print_recap()
        print(('n_iter: %4d | best_score: %2d | avg_score: %5.2f ' +
                '| alpha: %6.4f | q_dict_size: %3d') % (
                self.iter, self.best_score, self.sum_score/self.decay_rate,
                self.alpha, len(self.q_dict)
        ))

    # walls & snake_body & snake_pos
    def get_danger_vect(self):
        danger = [0 for i in range(4)]

        if self.snake_pos.x < 1:
            danger[0] = 1
        if self.snake_pos.x > self.size_x - 2:
            danger[2] = 1
        if self.snake_pos.y < 1:
            danger[1] = 1
        if self.snake_pos.y > self.size_y - 2:
            danger[3] = 1

        for block in self.snake_body[1:]:
            pos = self.snake_pos - block
            if pos == Direction.RIGHT.value:
                danger[0] = 1
            if pos == Direction.LEFT.value:
                danger[2] = 1
            if pos == Direction.UP.value:
                danger[1] = 1
            if pos == Direction.DOWN.value:
                danger[3] = 1

        return danger

    # food_pos & snake_pos
    def get_food_vect(self):
        food_arr = [0 for i in range(8)]
        if self.food_pos.x == self.snake_pos.x:
            if self.food_pos.y >= self.snake_pos.y:
                food_arr[0] = 1
            else:
                food_arr[4] = 1

        elif self.food_pos.x > self.snake_pos.x:
            if self.food_pos.y > self.snake_pos.y:
                food_arr[1] = 1
            elif self.food_pos.y == self.snake_pos.y:
                food_arr[2] = 1
            else:
                food_arr[3] = 1

        elif self.food_pos.x < self.snake_pos.x:
            if self.food_pos.y < self.snake_pos.y:
                food_arr[5] = 1
            elif self.food_pos.y == self.snake_pos.y:
                food_arr[6] = 1
            else:
                food_arr[7] = 1

        return food_arr

    # returns the corresponding value of the binary array
    def _hash_bin(self, array):
        hash = 0
        n = len(array)
        power = 1
        for i in range(n):
            hash += array[n-1-i] * power
            power *= 2
        return hash
