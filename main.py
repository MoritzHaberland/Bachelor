from proto_grasp.model.model_factory import ModelFactory
from config.neural_net_config import get_neural_net_config

if __name__ == "__main__":
    model_config = get_neural_net_config()
    factory = ModelFactory()
    model = factory.get_model("neural_net",model_config)


