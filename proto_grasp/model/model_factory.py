from proto_grasp.model.neural_net import NeuralNet

# Model factory to handle multiple models
class ModelFactory:
    def __init__(self):
        self.models = {
            'neural_net': NeuralNet,  
        }

    def get_model(self, model_config):
        if model_config.name not in self.models:
            raise ValueError(f"Model {model_config.name} not found.")
        return self.models[model_config.name](model_config)