from proto_grasp.model.model_factory import ModelFactory
from proto_grasp.data.dataset_factory import DatasetFactory
from proto_grasp.training.trainer import Trainer
from proto_grasp.testing.tester import tester
from config.config import get_config

import torch
import torch.nn as nn

def setup():
    # initialize configs
    config = get_config()
    model_config = config.model
    train_config = config.training

    # build model
    model_factory = ModelFactory()
    model = model_factory.get_model(model_config)

    # build training loaders
    data_factory = DatasetFactory()
    train_set = data_factory.get_dataset(train_config.dataset,train=True,download=True)
    test_set = data_factory.get_dataset(train_config.dataset,train=False,download=False)

    # build Trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)  
    trainer = Trainer(model, criterion, optimizer,train_set.get_dataloader(train_config.batch_size),
                      train_config.num_epochs)
    
    # train model
    trainer.train()

    # store model
    tester(test_set.get_dataloader(train_config.batch_size,shuffle=False), model)

if __name__ == "__main__":
    setup()
    

