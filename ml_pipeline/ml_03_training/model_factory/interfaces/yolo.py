from model_factory.interfaces.flow import FlowInterface
from abc import abstractmethod

class ObjectYoloInterface(FlowInterface):
    
    @abstractmethod
    def detect_and_predict(self):
        pass