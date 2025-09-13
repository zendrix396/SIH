from enum import Enum, auto

class EventType(Enum):
    TRAIN_DEPARTURE = auto()
    TRAIN_ARRIVAL = auto()
    SIMULATION_END = auto()

class Event:
    def __init__(self, event_type, time, data=None):
        self.type = event_type
        self.time = time
        self.data = data or {}

    def __lt__(self, other):
        return self.time < other.time
