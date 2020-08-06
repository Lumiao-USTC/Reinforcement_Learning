import abc


class basic_trainer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, train_data, **kwargs):
        pass

