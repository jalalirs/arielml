# arielml/utils/observable.py

from typing import List, Callable
from .signals import DetrendingProgress

class Observable:
    """Base class for objects that can emit progress signals."""
    
    def __init__(self):
        self._observers: List[Callable[[DetrendingProgress], None]] = []
        self._should_stop = False
    
    def add_observer(self, callback: Callable[[DetrendingProgress], None]):
        """Add an observer to receive progress signals."""
        if callback not in self._observers:
            self._observers.append(callback)
    
    def remove_observer(self, callback: Callable[[DetrendingProgress], None]):
        """Remove an observer."""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def notify_observers(self, signal: DetrendingProgress):
        """Notify all observers with a progress signal."""
        for observer in self._observers:
            try:
                observer(signal)
            except Exception as e:
                # Don't let observer errors break the pipeline
                print(f"Observer error: {e}")
    
    def request_stop(self):
        """Request that the current operation should stop."""
        self._should_stop = True
    
    def check_stop_request(self):
        """Check if a stop has been requested and raise an exception if so."""
        if self._should_stop:
            raise InterruptedError("Operation stopped by user")
    
    def reset_stop_flag(self):
        """Reset the stop flag for a new operation."""
        self._should_stop = False 