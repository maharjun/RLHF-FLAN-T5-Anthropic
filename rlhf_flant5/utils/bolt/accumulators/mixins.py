from abc import ABC, abstractmethod

class Loggable(ABC):
    @abstractmethod
    def log(self):
        ...
