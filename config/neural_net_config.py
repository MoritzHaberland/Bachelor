from ml_collections import config_dict

def get_neural_net_config():
    config = config_dict.ConfigDict()
    config.input_size = 784
    config.hidden_size = 500
    config.num_classes = 10
    return config


