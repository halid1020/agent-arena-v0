from agent_arena.agent.utilities.torch_utils import *


OPTIMISERS = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
}

ACTIVATIONS = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity,
    'tan': nn.Tanh
}



def build_mlp(layers, activation_function, dropout=0.0) -> nn.Sequential:
    mlp_layers = []
    for i in range(len(layers) - 1):
        mlp_layers.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:  # Don't add activation function to the output layer
            mlp_layers.append(ACTIVATIONS[activation_function]())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
    return nn.Sequential(*mlp_layers)