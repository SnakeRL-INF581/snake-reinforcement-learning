from utils.direction import Direction
import random
from utils.visual import Visual
from abc import ABC, abstractmethod

class ABCTrainer(ABC):

    def __init__(self, gamma, decay_rate, size_x, size_y, use_deep_learning):

        self.iter = 0

        # Reinforcement learning parameters
        self.gamma = gamma
        self.decay_rate = decay_rate

        # Game parameters
        self.size_x = size_x
        self.size_y = size_y

        # Training data
        self.reward = 0

        # Actual situation
        self.state = None
        self.action = None
        self.direction = None
        self.snake_pos = None
        self.snake_body = None
        self.food_pos = None

        # Scores
        self.sum_score = 0
        self.best_score = 0
        self.avg_scores = []
        self.best_scores = []

        # GFX
        self.visual = Visual(size_x, size_y)

        # deep or not
        self.use_deep_learning = use_deep_learning

    def _reinit(self):
        x_init = random.randrange(4, self.size_x-3)
        y_init = random.randrange(2, self.size_y-3)
        self.snake_pos = [x_init, y_init]
        self.snake_body = [[x_init, y_init], [x_init-1, y_init],
                            [x_init-2, y_init]]

        self.food_pos = [random.randrange(1, self.size_x-1),
                            random.randrange(1, self.size_y-1)]
        self.direction = Direction.RIGHT
        self.state = self.get_state()
        self.action = self.choose_action(self.state)
        self.iter += 1


    @abstractmethod
    def update_hyperparameters(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def choose_action(self, state):
        pass

    # To be implemented in SARSA and Q-learning train
    def update_q_dict(self, state, state2, action, action2):
        pass

    # To be implemented in DQL train
    def handle_deep_learning(self, state1, state2, action1, reward):
        pass

    @abstractmethod
    def print_recap(self):
        pass

    def save_data(self):
        self.avg_scores.append(self.sum_score/self.decay_rate)
        self.best_scores.append(self.best_score)

    def iterate(self, visual=False, speed=0, test=False, reward_plus=100,
                    reward_minus=-200, nb_step_max=250):
        if visual and speed == 0:
            raise Error("Usage: train.iterate(visual=True, speed=speed)")

        self._reinit()
        if self.iter % self.decay_rate == 0:
            self.print_recap()
            if not test:
                self.update_hyperparameters()
            self.save_data()
            self.best_score = 0
            self.sum_score = 0

        nb_step_wo_food = 0
        score = 0
        exit = False

        while not exit:
            nb_step_wo_food += 1

            if nb_step_wo_food >= nb_step_max:
                nb_step_wo_food = 0
                exit = True
                break

            new_state = self.get_state()
            new_action = self.choose_action(new_state)

            self.reward = 0

            # Make sure the snake cannot move in the opposite direction
            if not self.direction.is_opposite(new_action):
                self.direction = new_action

            # Move the snake
            self.direction.move(self.snake_pos)

            # Snake body growing mechanism
            self.snake_body.insert(0, list(self.snake_pos))
            if self.snake_pos == self.food_pos:
                score += 1
                nb_step_wo_food = 0
                self.reward = reward_plus
            else:
                self.snake_body.pop()

            # Spawning food on the screen
            if nb_step_wo_food == 0:
                self.food_pos = [random.randrange(1, self.size_x),
                    random.randrange(1, self.size_y)]

            # GFX
            if visual:
                self.visual.draw(self.snake_body, self.food_pos, score, speed)

            # Update data
            if not test:
                if self.use_deep_learning:
                    self.handle_deep_learning(self.state, new_state,
                                                self.action, self.reward)
                else:
                    self.update_q_dict(self.state, new_state,
                                        self.action, new_action)
            self.action = new_action
            self.state = new_state

            # Game Over conditions
            # - Getting out of bounds
            if (self.snake_pos[0] < 0 or self.snake_pos[0] > self.size_x-1 or
                    self.snake_pos[1] < 0 or self.snake_pos[1] > self.size_y-1):
                self.reward = reward_minus
                exit = True
                break

            # - Touching the snake body
            for block in self.snake_body[1:]:
                if self.snake_pos == block:
                    self.reward = reward_minus
                    nb_step_wo_food = 0
                    exit = True
                    break

        self.sum_score += score
        if score > self.best_score:
            self.best_score = score

        return score
