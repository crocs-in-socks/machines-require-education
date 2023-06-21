import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

font = pygame.font.Font('freesansbold.ttf', 32)

WHITE = (255, 255, 255)
RED = (255, 0, 0)
DARKRED = (100, 0, 0)
GREEN = (0, 255, 0)
DARKGREEN = (0, 100, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 420


class Direction(Enum):
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4


Point = namedtuple("Point", "x, y")


class GameAI:

    def __init__(self, w=1200, h=800):
        self.w = w
        self.h = h

        self.leftTurn = 0
        self.rightTurn = 0

        self.reward = 0

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snek")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w//2, self.h//2)

        # The body initially contains 3 blocks
        self.fullBody = [self.head,
                         Point(self.head.x - BLOCK_SIZE, self.head.y),
                         Point(self.head.x - 2*BLOCK_SIZE, self.head.y)
                         ]

        self.score = 0
        self.food = None
        self.placeFood()
        self.frameIteration = 0

    def placeFood(self, pt=None):

        if pt == None:
            x = random.randint(0, (self.w - BLOCK_SIZE) //
                               (BLOCK_SIZE)) * (BLOCK_SIZE)
            y = random.randint(0, (self.h - BLOCK_SIZE) //
                               (BLOCK_SIZE)) * (BLOCK_SIZE)
            self.food = Point(x, y)

            # If the food spawns inside the snake
            if self.food in self.fullBody:
                self.placeFood()
        else:
            x = self.head.x + 5*BLOCK_SIZE
            y = self.head.y
            self.food = Point(x, y)

    def playStep(self, action):

        self.frameIteration += 1
        # Take user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the snake
        self.move(action)
        # if (self.frameIteration % len(self.fullBody) == 0) and self.frameIteration > 50 * len(self.fullBody):
        #     reward = -1
        # else:
        #     reward = 0
        self.reward = 0
        # inserting updated head position to the front of snake
        self.fullBody.insert(0, self.head)

        gameOver = False
        if self.isColliding():
            gameOver = True
            return self.reward, gameOver, self.score

        if self.frameIteration > 400 * len(self.fullBody):
            self.reward = -10
            gameOver = True
            return self.reward, gameOver, self.score

        if self.head == self.food:
            self.score += 1
            self.reward = 10
            self.placeFood()
        else:
            # Remove the last block in snake
            self.fullBody.pop()

        # Update the UI and clock
        self.updateUI()
        self.clock.tick(SPEED)

        # Return game over and score
        return self.reward, gameOver, self.score

    def isColliding(self, pt=None):
        if pt == None:
            pt = self.head

        # hits a boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y < 0 or pt.y > self.h - BLOCK_SIZE:
            self.reward = -10
            return True
        # hits itself
        if pt in self.fullBody[1:]:
            self.reward = -10 * (len(self.fullBody) - 3)
            return True

        return False

    def isCollinear(self, pt):
        return (self.food.x - self.head.x) * (pt.y -
                                              self.head.y) == (pt.x - self.head.x) * (self.food.y - self.head.y)

    def isBetween(self, pt):
        return (self.head <= pt <= self.food or self.food <= pt <= self.head)

    def hasLineOfSight(self):
        for pt in self.fullBody[1:]:
            if self.isCollinear(pt) and self.isBetween(pt):
                return False
        return True

    def distanceXToFood(self):
        return abs(self.head.x - self.food.x)

    def distanceYToFood(self):
        return abs(self.head.y - self.food.y)

    def updateUI(self):
        self.display.fill(BLACK)

        for block in self.fullBody:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(
                block.x, block.y, BLOCK_SIZE, BLOCK_SIZE))
            # Fanciness
            # pygame.draw.rect(self.display, GREEN, pygame.Rect(
            #     block.x + 4, block.y + 4, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        # More fanciness
        # pygame.draw.rect(self.display, RED, pygame.Rect(
        #     self.food.x + 4, self.food.y + 4, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render(
            "Score = " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def checkTurns(self):
        self.leftTurn = self.leftTurn % 3
        self.rightTurn = self.rightTurn % 3

        return self.leftTurn, self.rightTurn

    def move(self, action):

        clockwise = [Direction.RIGHT, Direction.DOWN,
                     Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if action == [1, 0, 0]:
            # no change in direction
            newDirection = self.direction
            self.rightTurn = 0
            self.leftTurn = 0
        elif action == [0, 1, 0]:
            # turn right
            nextIdx = (idx + 1) % 4
            newDirection = clockwise[nextIdx]
            self.rightTurn += 1
            self.leftTurn = 0
        elif action == [0, 0, 1]:
            # turn left
            nextIdx = (idx - 1) % 4
            newDirection = clockwise[nextIdx]
            self.leftTurn += 1
            self.rightTurn = 0

        self.direction = newDirection

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            move_x = BLOCK_SIZE
            move_y = 0
        elif self.direction == Direction.LEFT:
            move_x = -1 * BLOCK_SIZE
            move_y = 0
        elif self.direction == Direction.UP:
            move_x = 0
            move_y = -1 * BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            move_x = 0
            move_y = BLOCK_SIZE

        x += move_x
        y += move_y
        self.head = Point(x, y)
