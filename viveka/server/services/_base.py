"""Base class for stateful mock services."""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    """Service-level error carrying a code (e.g. 'UPI:5012')."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class MockService:
    """Each service is a stateful Python class. Subclasses register `_op_<name>` handlers."""

    name: str = "abstract"

    def __init__(self) -> None:
        self.reset({})

    def reset(self, initial_state: dict[str, Any]) -> None:
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        raise NotImplementedError

    def execute(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_op_{operation}", None)
        if handler is None:
            raise ServiceError(
                code=f"{self.name.upper()}:UNKNOWN_OP",
                message=f"Unknown operation '{operation}' on {self.name}",
            )
        return handler(params or {})
