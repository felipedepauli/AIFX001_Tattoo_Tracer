from model_factory.interfaces.flow import FlowInterface
from abc import abstractmethod

class ObjectClassificationInterface(FlowInterface):

    @abstractmethod
    def predict(self):
        pass
