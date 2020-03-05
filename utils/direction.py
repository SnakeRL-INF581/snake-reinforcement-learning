import random
from enum import Enum

class Direction(Enum):
    LEFT  = 0
    UP    = 1
    RIGHT = 2
    DOWN  = 3

    def is_opposite(self, direction):
        return self.value == (direction.value + 2) % 4

    # Change pos by moving in the corresponding direction
    def move(self, pos):
        pos[self.value%2] += 2*(self.value//2)-1

    @staticmethod
    def random():
        return random.choice(list(Direction))
