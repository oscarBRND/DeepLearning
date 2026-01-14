import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def parameters_and_gradients(
        self
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:

        parameters: dict[str, np.ndarray] = {}
        gradients: dict[str, np.ndarray] = {}

        for i, layer in enumerate(self.layers):

            layer_params = layer.parameters()
            layer_grads = layer.gradients()

            # Skip layers without parameters (e.g. activations)
            if layer_params is not None:
                for name, value in layer_params.items():
                    parameters[f"layer{i}.{name}"] = value

            if layer_grads is not None:
                for name, value in layer_grads.items():
                    gradients[f"layer{i}.{name}"] = value

        return parameters, gradients

    
    def set_parameters(self, updated_parameters: dict[str, np.ndarray]) -> None:
        for i, layer in enumerate(self.layers):
            layer_params = layer.parameters()
            if layer_params is not None:
                for name in layer_params.keys():
                    layer_params[name] = updated_parameters[f"layer{i}.{name}"]
