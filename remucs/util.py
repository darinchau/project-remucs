from __future__ import annotations
from typing import Generic, TypeVar, Optional, Callable, Type

T = TypeVar('T')


class Result(Generic[T]):
    def __init__(self, value: T | None, success: bool, error: Optional[str] = None):
        self.value = value
        self.successful = success
        self.error = error

    @staticmethod
    def success(value: T) -> Result[T]:
        return Result(value, True)

    @staticmethod
    def failure(error: str) -> Result[T]:
        return Result(None, False, error)

    def __bool__(self):
        return self.successful

    def __str__(self):
        if self.successful:
            return f"Success: {self.value}"
        else:
            return f"Failure: {self.error}"

    def __repr__(self):
        return str(self)

    def unwrap(self) -> T:
        if not self.successful:
            raise ValueError(self.error)
        assert self.value is not None  # to make mypy happy
        return self.value
