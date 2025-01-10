from model_factory.interfaces.flow import FlowInterface
from abc import abstractmethod

class ObjectDetectionInterfaces(FlowInterface):

    @abstractmethod
    def detect(self):
        pass