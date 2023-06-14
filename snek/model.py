import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class linearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def saveModel(self, fileName='model.pth'):
        modelFolderPath = "./model"
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        fileName = os.path.join(modelFolderPath, fileName)
        torch.save(self.state_dict(), fileName)


class QTrainer:
    def __init__(self, model, learningRate, gamma):
        self.model = model
        self.learningRate = learningRate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        self.lossFunction = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, gameOver):
        state = torch.tensor(state, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # In order to give the tensor dimensions (1, x)
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # To make sure everything is in the correct shape
            gameOver = (gameOver, )

        # To obtain predicted Quality values with the current state
        prediction = self.model(state)

        # reward + gamma * max(next predicted Quality values)

        target = prediction.clone()

        # Iterating over all the examples
        for idx in range(len(gameOver)):
            newQuality = reward[idx]
            if not gameOver[idx]:
                newQuality = reward[idx] + self.gamma * \
                    torch.max(self.model(nextState[idx]))

            # From what I understand, we're replacing the value of the highest predicted quality from the current state with the new predicted quality (as that is the path we take ?)
            target[idx][torch.argmax(action).item()] = newQuality

            # Applying the loss function

        self.optimizer.zero_grad()  # necessary for some reason in PyTorch

        # loss = square(newQuality - Quality)
        loss = self.lossFunction(target, prediction)
        # Applying backprop to calculate ideal changes in weights and biases
        loss.backward()

        # Updating the values of the weights and biases
        self.optimizer.step()

        # From what I understand further, we first choose an action to take based on the current state (this is prediction), then using the formula we calculate the quality of the nextState which is a resultant of the currentState. Then, we find how wrong/correct the action we took was using the loss function, and then we update our weights and biases as required.
