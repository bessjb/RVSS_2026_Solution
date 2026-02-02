import concurrent.futures
import time

from enum import Enum, auto

import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 5)
        self.relu = nn.ReLU()

        self.layers = [
            self.conv1,
            self.conv2,
            self.pool,
            self.fc1,
            self.fc2,
            self.relu
        ]

    def transform_images(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((40, 60)),
                                        transforms.Normalize(
                                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])

    def forward(self, x):
        # extract features with convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        # linear layer for classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class NeuralNetWrapper:
    class NNState(Enum):
        # No Inference results, ready for request
        READY = auto()
        # Processing inference results
        INPROGRESS = auto()
        # Inference has completed, ready for retrieval
        COMPLETE = auto()

    def __init__(self, executor):
        self.state = self.NNState.READY
        self.executor = executor

    @classmethod
    def load_model(self, executor, model_path, weights_only=True):
        self.__init__(self, executor)
        self.neural_network = NeuralNet()
        print(self.neural_network.layers[0].weight)
        self.neural_network.load_state_dict(torch.load(model_path))
        print(self.neural_network.layers[0].weight)

    def run_neural_network(self, image):

        self.neural_net.transform_images(image)
        self.neural_net.forward(image)
        time.sleep(0.1)
        return 5

    def run_inference(self, image):
        ret_result = None
        if self.state is self.NNState.READY:
            if image is not None:
                self.network_future = self.executor.submit(
                    self.run_neural_network, image)
                self.state = self.NNState.INPROGRESS
        elif self.state is self.NNState.INPROGRESS:
            if self.network_future.done():
                self.state = self.NNState.COMPLETE
        elif self.state is self.NNState.COMPLETE:
            ret_result = self.network_future.result()
            self.state = self.NNState.READY
        return ret_result
