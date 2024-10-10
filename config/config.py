from ml_collections import config_dict
from config.neural_net_config import get_neural_net_config
from config.training_config import get_training_config

def get_config():
    config = config_dict.ConfigDict()

    # Allgemeine Einstellungen
    config.project_name = 'my_model_project'
    config.run_name = 'experiment_1'
    
    # Füge separate Konfigurationen hinzu
    config.models = get_neural_net_config()
    config.training = get_training_config()

    return config
