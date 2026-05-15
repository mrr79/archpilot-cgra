"""
Exception module for the ArchPilot-CGRA simulator.

Complies with: SYRS-FUN-010, SYRS-MNT-001, SYRS-MNT-002, SYRS-QLY-002
"""

from typing import Any


class SimulationException(Exception):
    """Base exception for all ArchPilot-CGRA simulator errors."""


class OverflowSimulationException(SimulationException):
    """
    Raised when an operation result exceeds the representable range
    for the configured bit width of the PE.

    Complies with SYRS-FUN-010.

    Attributes:
        pe_id (tuple[int, int]):   identifier (row, column) of the PE.
        value (int):               value that caused the overflow.
        max_value (int):           maximum allowed value for data_width bits.
        pe_state (dict[str, Any]): snapshot of the PE state at the time
                                   of the error.
    """

    def __init__(
        self,
        pe_id: tuple[int, int],
        value: int,
        max_value: int,
        pe_state: dict[str, Any],
    ) -> None:
        self.pe_id: tuple[int, int] = pe_id
        self.value: int = value
        self.max_value: int = max_value
        self.pe_state: dict[str, Any] = pe_state
        super().__init__(
            f"Overflow in PE{pe_id}: value={value} exceeds "
            f"max={max_value}. State: {pe_state}"
        )


class InvalidOperationException(SimulationException):
    """
    Raised when the Functional Unit receives an unsupported opcode.

    Attributes:
        opcode (str): the invalid opcode that triggered the error.
    """

    def __init__(self, opcode: str) -> None:
        self.opcode: str = opcode
        super().__init__(f"Unsupported operation: '{opcode}'")


class ConfigMemoryException(SimulationException):
    """
    Raised when an invalid configuration is loaded into the Config Memory.

    Typical causes: pe_id outside mesh bounds, malformed instruction dicts.
    """
