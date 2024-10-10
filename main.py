from proto_grasp.model.model_factory import ModelFactory
from proto_grasp.data.dataset_factory import DatasetFactory
from config.config import get_config

if __name__ == "__main__":
    # initialize configs
    config = get_config()
    model_config = config.model
    train_config = config.training

    #build model
    model_factory = ModelFactory()
    model = model_factory.get_model(model_config)

    #build training loaders
    data_factory = DatasetFactory()
    dataset = data_factory.get_dataset(train_config.dataset)


