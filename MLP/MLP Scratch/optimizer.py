# You are not allowed to import any other libraries or modules.

import torch

class SGD:
    def __init__(self, params, learning_rate):
        """
        Initialize the Stochastic Gradient Descent (SGD) optimizer.
        
        Args:
            params: List of model parameters to be updated.
            learning_rate: The learning rate for updating the parameters.
        """
        self.params = list(params)  # Model parameters to be updated
        self.learning_rate = learning_rate

    def step(self):
        """
        Perform a parameter update using Stochastic Gradient Descent (SGD).
        """
        #TODO: Implement the parameter update rule for SGD. Take these steps:
        # Loop over all parameters (self.params).
        for param in self.params:
        # Update each parameter using the gradient and learning rate.
            if param.grad is not None:
                param.data -= self.learning_rate * param.grad.data
        # Ensure that gradients are not being tracked during this operation.

    def zero_grad(self):
        """
        Zero the gradients for all parameters.
        """
        #TODO: Implement gradient zeroing for all parameters. Take these steps:
        # Loop over all parameters (self.params).
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
        # Zero out the gradient for each parameter.


