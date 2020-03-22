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

class ExtendedDirection(Enum):
    LEFT      = 0
    UPLEFT    = 1
    UP        = 2
    UPRIGHT   = 3
    RIGHT     = 4
    DOWNRIGHT = 5
    DOWN      = 6
    DOWNLEFT  = 7

    def cast(self):
        if self.value == 0:
            return Direction.LEFT
        if self.value == 2:
            return Direction.UP
        if self.value == 4:
            return Direction.RIGHT
        if self.value == 6:
            return Direction.DOWN

    @staticmethod
    def random():
        return random.choice(list(ExtendedDirection))