from abc import ABC, abstractmethod

class Transform(ABC):
    @abstractmethod
    def __init__(self, params):
        pass

    @abstractmethod
    def __call__(self, data):
        pass

    @abstractmethod
    def postprocess(self, data):
        pass