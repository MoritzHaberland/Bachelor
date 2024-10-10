from ml_collections import config_dict

def get_training_config():
    config = config_dict.ConfigDict()

    config.dataset = 'MNIST'
    config.num_epochs = 2
    config.batch_size = 100
    config.learning_rate = 0.001

    return config
