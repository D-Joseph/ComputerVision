

import torch
import torch.nn.functional as F
import torch.nn as nn


class autoencoderMLP4Layer(nn.Module):

	def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
		""" Initialize the layers of the neural network. """
		super().__init__()
		half_input = N_input // 2
		self.fc1 = nn.Linear(N_input, half_input)
		self.fc2 = nn.Linear(half_input, N_bottleneck)
		self.fc3 = nn.Linear(N_bottleneck, half_input)
		self.fc4 = nn.Linear(half_input, N_output)
		self.type = "MLP4"
		self.input_shape = (1, 28*28)

	def encode(self, input_tensor):
		X = self.fc1(input_tensor)
		X = F.relu(X)
		X = self.fc2(X)
		X = F.relu(X)
		return X

	def decode(self, bottleneck_tensor):
		X = self.fc3(bottleneck_tensor)
		X = F.relu(X)
		X = self.fc4(X)
		X = torch.sigmoid(X)
		return X

	def forward(self, X):
		return self.decode(self.encode(X))