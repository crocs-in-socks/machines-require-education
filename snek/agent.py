import torch
import random
import numpy as np
from collections import deque
from game import GameAI, Direction, Point
from model import linearQNet, QTrainer
# from plotter import plot

learningRate = 0.003
MAX_MEMORY = 100000
BATCH_SIZE = 1000


class Agent:
    def __init__(self):
        self.gameNumber = 0
        self.epsilon = 0  # controls randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        # model, trainer
        self.model = linearQNet(14, 300, 3)
        self.trainer = QTrainer(self.model, learningRate, self.gamma)

    def getState(self, game):
        head = game.head
        blockToLeft = Point(head.x - 10, head.y)
        blockToRight = Point(head.x + 10, head.y)
        blockToUp = Point(head.x, head.y - 10)
        blockToDown = Point(head.x, head.y + 10)

        hasLineOfSight = game.hasLineOfSight()
        distanceXToFood = game.distanceXToFood()
        distanceYToFood = game.distanceYToFood()
        self.consecutiveLeft, self.consecutiveRight = game.checkTurns()

        directionIsLeft = (game.direction == Direction.LEFT)
        directionIsRight = (game.direction == Direction.RIGHT)
        directionIsUp = (game.direction == Direction.UP)
        directionIsDown = (game.direction == Direction.DOWN)

        dangerAhead = (game.isColliding(blockToLeft) and directionIsLeft) or (game.isColliding(blockToRight) and directionIsRight) or (
            game.isColliding(blockToUp) and directionIsUp) or (game.isColliding(blockToDown) and directionIsDown)

        dangerToLeft = (game.isColliding(blockToDown) and directionIsLeft) or (game.isColliding(blockToUp) and directionIsRight) or (
            game.isColliding(blockToLeft) and directionIsUp) or (game.isColliding(blockToRight) and directionIsDown)

        dangerToRight = (game.isColliding(blockToUp) and directionIsLeft) or (game.isColliding(blockToDown) and directionIsRight) or (
            game.isColliding(blockToRight) and directionIsUp) or (game.isColliding(blockToLeft) and directionIsDown)

        foodToLeft = game.food.x < head.x
        foodToRight = game.food.x > head.x
        foodToUp = game.food.y < head.y
        foodToDown = game.food.y > head.y

        state = [
            hasLineOfSight,
            distanceXToFood,
            distanceYToFood,
            dangerAhead,
            dangerToRight,
            dangerToLeft,
            directionIsLeft,
            directionIsRight,
            directionIsUp,
            directionIsDown,
            foodToLeft,
            foodToRight,
            foodToUp,
            foodToDown
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, nextState, gameOver):
        self.memory.append((state, action, reward, nextState, gameOver))

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            smallSample = random.sample(self.memory, BATCH_SIZE)
        else:
            smallSample = self.memory

        states, actions, rewards, nextStates, gameOvers = zip(*smallSample)
        # zip combines all elements at the same indexes over a bunch of lists into tuples

        self.trainer.trainStep(states, actions, rewards, nextStates, gameOvers)

    def trainShortMemory(self, state, action, reward, nextState, gameOver):
        self.trainer.trainStep(state, action, reward, nextState, gameOver)

    def getAction(self, state):

        # trade-off between random moves and model predicted moves
        self.epsilon = 80 - self.gameNumber
        nextMove = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            moveIdx = random.randint(0, 2)
            nextMove[moveIdx] = 1

            if self.consecutiveLeft >= 2 and nextMove == [0, 0, 1]:
                moveIdx = random.randint(0, 1)
                nextMove = [0, 0, 0]
                nextMove[moveIdx] = 1
            elif self.consecutiveRight >= 2 and nextMove == [0, 1, 0]:
                moveIdx = random.randint(0, 1)
                nextMove = [0, 0, 0]
                if moveIdx == 1:
                    nextMove[2] = 1
                else:
                    nextMove[0] = 1
        else:
            # converting the currentState to a tensor to feed into our model, and also changing data type to float for more precision
            currentState = torch.tensor(state, dtype=torch.float)
            prediction = self.model(currentState)

            # argmax() finds the maximum value along a specified axis of a tensor
            # prediction is a tensor that is returned by the model

            # .item() turns the tensor with one element returned by argmax() to a number
            moveIdx = torch.argmax(prediction).item()
            nextMove[moveIdx] = 1

        if self.consecutiveLeft >= 2 and nextMove == [0, 0, 1]:
            moveIdx = random.randint(0, 1)
            nextMove = [0, 0, 0]
            nextMove[moveIdx] = 1
        elif self.consecutiveRight >= 2 and nextMove == [0, 1, 0]:
            moveIdx = random.randint(0, 1)
            nextMove = [0, 0, 0]
            if moveIdx == 1:
                nextMove[2] = 1
            else:
                nextMove[0] = 1

        return nextMove


def train():
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = 0
    agent = Agent()
    game = GameAI()

    while True:
        # get current state
        currentState = agent.getState(game)

        # get next move based on current state
        nextMove = agent.getAction(currentState)

        # perform the next move
        reward, gameOver, score = game.playStep(nextMove)

        # get state after move
        newState = agent.getState(game)

        # train short memory
        agent.trainShortMemory(currentState, nextMove,
                               reward, newState, gameOver)

        # remember
        agent.remember(currentState, nextMove,
                       reward, newState, gameOver)

        if gameOver == True:
            # train long / replay / experience memory
            game.reset()
            agent.gameNumber += 1
            agent.trainLongMemory()
            print("Trained long memory")

            if score > record:
                record = score
                agent.model.saveModel()

            print("Game #", agent.gameNumber,
                  "Score =", score, "Record =", record)

            # plotScores.append(score)
            # totalScore += score
            # meanScore = totalScore / agent.gameNumber
            # plotMeanScores.append(meanScore)
            # plot(plotScores, plotMeanScores)


if __name__ == "__main__":
    train()
