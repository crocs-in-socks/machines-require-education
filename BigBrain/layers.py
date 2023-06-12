import numpy as np
np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros((points*classes), dtype='uint8')
    for classNumber in range(classes):
        ix = range(points*classNumber, points*(classNumber+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(classNumber*4, (classNumber+1)*4, points) * np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = classNumber
    return X, y

class DenseLayer:
    def __init__(self, nInputs, nNeurons):
        # We do it this way, so that we don't have to transpose later
        self.weights = np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class reluActivation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class softmaxActivation:
    def forward(self, inputs):
        # To prevent overflow
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss
    
class CategoricalCrossEntropyLoss(Loss):
    def forward(self, y_pred, y_train):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # y_train is scalar values
        if len(y_train.shape) == 1:
            correctConfidences = y_pred_clipped[range(samples), y_train]
        else:
            # y_train is one-hot encoded values
            correctConfidences = np.sum(y_pred_clipped * y_train, axis=1)
        
        negativeLogLikelihoods = - np.log(correctConfidences)
        return negativeLogLikelihoods

X, y = create_data(100, 3)

layer1 = DenseLayer(2, 3)
activation1 = reluActivation()

layer2 = DenseLayer(3, 3)
activation2 = softmaxActivation()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])

lossFunction = CategoricalCrossEntropyLoss()
loss = lossFunction.calculate(activation2.output, y)

print("Loss = ", loss)