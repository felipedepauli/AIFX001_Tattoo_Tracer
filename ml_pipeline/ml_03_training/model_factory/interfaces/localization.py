from model_factory.interfaces.flow import FlowInterface
from abc import abstractmethod

class ObjectLocaterInterface(FlowInterface):
    
    @abstractmethod
    def locate(self):
        pass