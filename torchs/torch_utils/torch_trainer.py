import abc
from trainer.trainer import basicTrainer
from torchs.torch_utils.numpy_to_torch import torch_train_data_from_numpy


class torchTrainer(basicTrainer):
    def train(self, train_data):
        train_data = torch_train_data_from_numpy(train_data)
        self.torch_train(train_data)

    @abc.abstractmethod
    def torch_train(self, train_data):
        pass
