import numpy as np
import torch


# Custom Pytorch Module for Neural Networks
class PyTorchNeuralNetwork(torch.nn.Module):
    """
    Initializes a PyTorch neural network model with specified number of inputs, outputs, and neurons.

    Parameters
    ----------
    n_inputs (int): Number of input features.
    n_outputs (int): Number of output classes.
    n_neurons (int): Number of neurons in hidden layers.

    Returns
    -------
    PyTorchNeuralNetwork.

    Raises
    -------
    None.
    """

    # Constructor
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_neurons,
        batch_size=1000,
        epochs=1,
        learning_rate=0.001,
    ):
        super(PyTorchNeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(n_inputs, n_neurons)
        self.fc2 = torch.nn.Linear(n_neurons, n_neurons)
        self.fc3 = torch.nn.Linear(n_neurons, n_outputs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    # Predictions
    def forward(self, x):
        """
        Performs the forward pass of the neural network.

        Parameters
        -------
        x (torch.Tensor): Input tensor to the neural network.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        y_pred = torch.nn.functional.softmax(self.fc3(x))
        return y_pred

    # Adding extra parameters for training
    def fit(self, x_train, y_train):
        """
        Fits the neural network to the training data.

        Parameters
        ----------
        x_train (array-like): Input training data.
        y_train (array-like): Target training data.

        Returns
        -------
        PyTorchNeuralNetwork: Trained neural network instance.

        Raises
        ------
        None.
        """
        x_train_tensor = torch.from_numpy(np.array(x_train).astype(np.float32))
        y_train_tensor = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # defining the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # defining Cross-Entropy loss
        criterion = torch.nn.NLLLoss()

        epochs = self.epochs
        for _ in range(epochs):
            for i, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(data)
                output = torch.log(output)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

        return self

    def predict(self, test):
        """
        Predicts using the trained neural network.

        Parameters
        -------
        test (torch.Tensor): Input tensor for prediction.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        self.eval()
        y_train_pred = []
        with torch.no_grad():
            output = self(test)

            y_train_pred.extend(output)

        y_train_pred = torch.stack(y_train_pred)
        return y_train_pred


# Custom Pytorch Module for Logistic Regression
class PyTorchLogisticRegression(torch.nn.Module):
    """
    Initializes a PyTorch logistic regression linear model with specified number of inputs, outputs.

    Parameters
    ----------
      n_inputs (int): Number of input features.
      n_outputs (int): Number of output classes.

    Returns
    -------
    PyTorchLogisticRegression.

    Raises
    -------
    None.
    """

    # Constructor
    def __init__(
        self, n_inputs, n_outputs, batch_size=1000, epochs=1, learning_rate=0.001
    ):
        super(PyTorchLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    # Predictions
    def forward(self, x):
        """
        Performs the forward pass of the logistic regression model.

        Parameters
        -------
        x (torch.Tensor): Input tensor to the logistic regression model.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        y_pred = torch.nn.functional.softmax(self.linear(x))
        return y_pred

    def fit(self, x_train, y_train):
        """
        Fits the logistic regression model to the training data.

        Parameters
        ----------
        x_train (array-like): Input training data.
        y_train (array-like): Target training data.
        batch_size (int): Size of each training batch.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.

        Returns
        -------
        PyTorchLogisticRegression: Trained logistic regression model instance.

        Raises
        ------
        None.
        """
        x_train_tensor = torch.from_numpy(np.array(x_train).astype(np.float32))
        y_train_tensor = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Defining the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Defining Cross-Entropy loss
        criterion = torch.nn.NLLLoss()

        epochs = self.epochs
        for _ in range(epochs):
            for i, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(data)
                output = torch.log(output)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

        return self

    def predict(self, test):
        """
        Predicts using the trained logistic regression model.

        Parameters
        -------
        test (torch.Tensor): Input tensor for prediction.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        self.eval()
        y_train_pred = []
        with torch.no_grad():
            output = self(test)

            y_train_pred.extend(output)

        y_train_pred = torch.stack(y_train_pred)
        return y_train_pred


# TensorFlow-based models have been deprecated.
class TensorflowNeuralNetwork:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "TensorflowNeuralNetwork is disabled because TensorFlow support was removed."
        )


class TensorflowLogisticRegression:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "TensorflowLogisticRegression is disabled because TensorFlow support was removed."
        )
