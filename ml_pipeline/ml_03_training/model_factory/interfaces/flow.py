from abc import ABC, abstractmethod

class FlowInterface(ABC):
    @abstractmethod
    def generate_conf(self):
        pass

    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def generate_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def export_model(self):
        pass