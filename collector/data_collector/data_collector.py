import abc


class basic_data_collector(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def collect_data(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_data(self, **kwargs):
        pass

    @abc.abstractmethod
    def end_collect(self):
        pass

