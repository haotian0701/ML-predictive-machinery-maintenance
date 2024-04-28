from typing import List
import torch
import torch.nn as nn


def make_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    hidden_activation: nn.Module = nn.ReLU,
    dropout: float = 0.0
) -> nn.Sequential:
    """
    Purpose: Constructs an MLP using specified parameters.
    :param input_dim: int representing the dimensionality of input features.
    :param hidden_dims: List[int] representing total neurons per hidden layer.
    :param output_dim: int representing the number of neurons in output layer.
    :param hidden_activation: nn.Module representing the activation function 
        used after each hidden layer.
    :param dropout: float representing the probability of an element to be zeroed
        (i.e., the dropout rate) after each hidden layer for regularization.
    :return: nn.Sequential representing a fully constructed MLP model.
    """
    num_hidden_layers = len(hidden_dims)
    seq_list = []

    if num_hidden_layers > 0:

        # initialize first hidden layer with input dimension and the first element of hidden_dims
        seq_list.append(nn.Linear(input_dim, hidden_dims[0]))
        seq_list.append(hidden_activation())

        # add dropout after first hidden layer if dropout rate is non-zero
        if dropout > 0.0:
            seq_list.append(nn.Dropout(dropout))

        for i in range(1, num_hidden_layers):

            # add subsequent hidden layers connecting each pair of consecutive dimensions in hidden_dims
            seq_list.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            seq_list.append(hidden_activation())

            if dropout > 0.0:
                # add dropout after each hidden layer if dropout rate is non-zero
                seq_list.append(nn.Dropout(dropout))

        # add output layer connected to last hidden layer
        seq_list.append(nn.Linear(hidden_dims[-1], output_dim))
    else:
        # if there are no hidden layers, directly connect input and output layers
        seq_list.append(nn.Linear(input_dim, output_dim))

    # assemble all components into sequential model
    mlp_model = nn.Sequential(*seq_list)

    return mlp_model


class RNN_Model(nn.Module):
    """
    A class with an RNN model which processes sequences using an RNN layer,
    followed by a customizable MLP classifier.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        classifier_mlp: nn.Module
    ) -> None:
        """
        Purpose: Initializes the RNN model with specified attributes and an MLP classifier.
        :param input_size: int representing the number of input features per time step.
        :param hidden_size: int representing the size of the RNN hidden state.
        :param num_layers: int representing the number of stacked RNN layers.
        :param dropout: float representing the dropout rate between RNN layers (not including last layer).
        :param classifier_mlp: nn.Module representing the MLP used for final classification.
        :return: None.
        """
        # initialize parent class, nn.Module
        super(RNN_Model, self).__init__()

        # set RNN attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # instantiate RNN
        self.rnn = nn.RNN(
            **{
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            }
        )

        # set classifier MLP attribute
        assert classifier_mlp is not None, \
            "Classifier MLP cannot be None!"
        self.classifier_mlp = classifier_mlp

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Purpose: Defines computation performed at each call of the RNN model.
        :param x: torch.Tensor representing input sequence to the RNN.
        :return: torch.Tensor representing output predictions from the MLP classifier.
        """
        # initialize hidden state for first input with zeros
        h0 = torch.zeros(
            size = (self.num_layers, x.size(1), self.hidden_size),
            device = x.device
        ).requires_grad_()

        # forward propagation
        _, rnn_output = self.rnn(
            input = x,
            hx = h0.detach()
        )

        # obtain last layer of last hidden state
        final_hidden_state = rnn_output[-1, :, :]

        # obtain last logits
        classifier_logits = self.classifier_mlp(
            final_hidden_state
        )

        return classifier_logits


class LSTM_Model(nn.Module):
    """
    A class with an LSTM model which processes sequences using a LSTM layer,
    followed by a customizable MLP classifier.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        classifier_mlp: nn.Module
    ) -> None:
        """
        Purpose: Initializes the LSTM model with specified attributes and an MLP classifier.
        :param input_size: int representing the number of input features per time step.
        :param hidden_size: int representing the size of the LSTM hidden state.
        :param num_layers: int representing the number of stacked LSTM layers.
        :param dropout: float representing the dropout rate between LSTM layers (not including last layer).
        :param classifier_mlp: nn.Module representing the MLP used for final classification.
        :return: None.
        """
        # initialize parent class, nn.Module
        super(LSTM_Model, self).__init__()

        # set LSTM attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # instantiate LSTM
        self.rnn = nn.LSTM(
            **{
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            }
        )

        # set classifier MLP attribute
        assert classifier_mlp is not None, \
            "Classifier MLP cannot be None!"
        self.classifier_mlp = classifier_mlp

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Purpose: Defines computation performed at each call of the LSTM model.
        :param x: torch.Tensor representing input sequence to the LSTM.
        :return: torch.Tensor representing output predictions from the MLP classifier.
        """
        # initialize hidden state for first input with zeros
        h0 = torch.zeros(
            size = (self.num_layers, x.size(1), self.hidden_size),
            device = x.device
        ).requires_grad_()

        # initialize cell state for first input with zeros
        c0 = torch.zeros(
            size = (self.num_layers, x.size(1), self.hidden_size),
            device = x.device
        ).requires_grad_()

        # forward propagation
        _, (rnn_output, _) = self.rnn(
            input = x,
            hx = (h0.detach(), c0.detach())
        )

        # obtain last layer of last hidden state
        final_hidden_state = rnn_output[-1, :, :]

        # obtain last logits
        classifier_logits = self.classifier_mlp(
            final_hidden_state
        )

        return classifier_logits


class GRU_Model(nn.Module):
    """
    A class with a GRU model which processes sequences using a GRU layer,
    followed by a customizable MLP classifier.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        classifier_mlp: nn.Module
    ) -> None:
        """
        Purpose: Initializes the GRU model with specified attributes and an MLP classifier.
        :param input_size: int representing the number of input features per time step.
        :param hidden_size: int representing the size of the GRU hidden state.
        :param num_layers: int representing the number of stacked GRU layers.
        :param dropout: float representing the dropout rate between GRU layers (not including last layer).
        :param classifier_mlp: nn.Module representing the MLP used for final classification.
        :return: None.
        """
        # initialize parent class, nn.Module
        super(GRU_Model, self).__init__()

        # set GRU attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # instantiate GRU
        self.rnn = nn.GRU(
            **{
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            }
        )

        # set classifier MLP attribute
        assert classifier_mlp is not None, \
            "Classifier MLP cannot be None!"
        self.classifier_mlp = classifier_mlp

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Purpose: Defines computation performed at each call of the GRU model.
        :param x: torch.Tensor representing input sequence to the GRU.
        :return: torch.Tensor representing output predictions from the MLP classifier.
        """
        # initialize hidden state for first input with zeros
        h0 = torch.zeros(
            size = (self.num_layers, x.size(1), self.hidden_size),
            device = x.device
        ).requires_grad_()

        # forward propagation
        _, rnn_output = self.rnn(
            input = x,
            hx = h0.detach()
        )

        # obtain last layer of last hidden state
        final_hidden_state = rnn_output[-1, :, :]

        # obtain last logits
        classifier_logits = self.classifier_mlp(
            final_hidden_state
        )

        return classifier_logits
