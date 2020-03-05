import torch
import numpy as np
import random
from utils.direction import Direction
from trainers.abctrainer import ABCTrainer
from utils.deep_learning_models import *

class DQLTrainer(ABCTrainer):

    def __init__(self, epsilon, epsilon_decay, learning_rate,
                    gamma, decay_rate, size_x, size_y,
                    batch_size=64, memory_size=10000, save_weight_frequency=10,
                    load_model_path=""):

        super().__init__(gamma, decay_rate, size_x, size_y, True)

        # Training data
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # NN init
        self.device = torch.device("cuda:0"
                        if torch.cuda.is_available() else "cpu")
        self.policy_net = Net()
        self.target_net = Net()
        if len(load_model_path) > 0:
            self.policy_net.load_state_dict(torch.load(load_model_path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                            lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        self.memory = ReplayMemory(memory_size)
        self.save_weight_frequency = save_weight_frequency
        self.batch_size = batch_size

        print(self.policy_net)

    # Implement abstract method
    def update_hyperparameters(self):
        super().update_hyperparameters()
        self.epsilon *= self.epsilon_decay

    # Implement abstract method
    def get_state(self):
        super().get_state()
        return torch.tensor(np.concatenate((
            self.get_environment_mat().reshape(1,-1),
            self.get_food_vect().reshape(1,-1)), axis=1))

    # Implement abstract method
    def choose_action(self, state):
        super().choose_action(state)
        random = np.random.rand()
        if random < (1 - self.epsilon):
            state = state.to(self.device)
            values, indices = self.policy_net(
                                state[0][:49], state[0][49:]).max(0)
            action = Direction(int(indices.numpy()))
        else:
            action = Direction.random()
        return action

    # Implement abstract method
    def handle_deep_learning(self, state1, state2, action1, reward):
        super().handle_deep_learning(state1, state2, action1, reward)
        action1 = torch.tensor([action1.value])
        reward = torch.tensor([reward])
        self.memory.push(state1, action1, state2, reward)

        self.optimize_model()

        if self.iter % self.save_weight_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # Implement abstract method
    def print_recap(self):
        super().print_recap()
        print(('n_iter: %4d | best_score: %2d | avg_score: %5.2f ' +
                '| epsilon: %6.4f | learning_rate: %6.4f') % (
                self.iter, self.best_score, self.sum_score/self.decay_rate,
                self.epsilon, self.learning_rate
        ))

    def get_environment_mat(self):
        env_size = 7
        danger = np.zeros((env_size,env_size))

        for j in range(-int(env_size/2), int((env_size + 1)/ 2)):
            for i in range(-int(env_size/2), int((env_size + 1)/ 2)):
                cell = [self.snake_pos[0] + j, self.snake_pos[1] + i]
                if self.in_danger(cell):
                    danger[j + env_size // 2, i + env_size // 2] = 1

        return danger

    def in_danger(self, cell):
        if (cell[0] < 0 or cell[0] > self.size_x-1 or
                cell[1] < 0 or cell[1] > self.size_y-1):
            return True

        for block in self.snake_body:
            if cell[0] == block[0] and cell[1] == block[1]:
                return True

    # food_pos & snake_pos
    def get_food_vect(self):
        food_arr = [0 for i in range(8)]
        if self.food_pos[0] == self.snake_pos[0]:
            if self.food_pos[1] >= self.snake_pos[1]:
                food_arr[0] = 1
            else:
                food_arr[4] = 1

        elif self.food_pos[0] > self.snake_pos[0]:
            if self.food_pos[1] > self.snake_pos[1]:
                food_arr[1] = 1
            elif self.food_pos[1] == self.snake_pos[1]:
                food_arr[2] = 1
            else:
                food_arr[3] = 1

        elif self.food_pos[0] < self.snake_pos[0]:
            if self.food_pos[1] < self.snake_pos[1]:
                food_arr[5] = 1
            elif self.food_pos[1] == self.snake_pos[1]:
                food_arr[6] = 1
            else:
                food_arr[7] = 1

        return np.array(food_arr)


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        reward_batch = torch.cat(batch.reward).to(self.device)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        expected_values = (self.target_net(
                            next_state_batch[:, :49],
                            next_state_batch[:, 49:]).max(1)[0] * self.gamma
                                + reward_batch)

        indices = action_batch.squeeze()
        output = self.policy_net(state_batch[:, :49], state_batch[:, 49:]
                    )[: , indices][:,0]

        # Optimize the model
        loss = self.loss_fn(output[0], expected_values[0])
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
