"""Live Person 1 sensor generation connected to Person 2 inference."""

__all__ = ["ReadingService"]


def __getattr__(name):
    if name == "ReadingService":
        from .service import ReadingService
        return ReadingService
    raise AttributeError(name)
