"""
Register File (RF) module for the Processing Element.

The Register File provides local data storage for each PE.
Its depth is independently configurable per cell within the
range defined by the architecture standard.

Complies with: SYRS-FUN-004, SYRS-MNT-001, SYRS-MNT-002, SYRS-QLY-002
"""

from typing import Final

from archpilot_cgra.exceptions import SimulationException

# Depth limits per SYRS-FUN-004.
# Final prevents accidental reassignment of these module-level constants.
RF_MIN_DEPTH: Final[int] = 2
RF_MAX_DEPTH: Final[int] = 16


class RegisterFile:
    """
    Register File (RF) of a CGRA Processing Element.

    Provides integer storage with configurable depth between 2 and 16
    registers per cell (SYRS-FUN-004). All registers are initialized to 0.
    Supports indexed read and write with range validation on every access.

    __slots__ is declared to eliminate the per-instance __dict__,
    saving ~200-400 bytes per object. In a 16×16 mesh this removes
    256+ extra dicts with zero behavioral change.

    Attributes:
        depth (int):      number of available registers [2, 16].
        data_width (int): bit width of each register.

    Example:
        >>> rf = RegisterFile(depth=4, data_width=32)
        >>> rf.write(0, 42)
        >>> rf.read(0)
        42
        >>> len(rf)
        4
    """

    __slots__ = ("depth", "data_width", "_registers")

    def __init__(self, depth: int = 4, data_width: int = 32) -> None:
        """
        Initializes the Register File.

        Args:
            depth (int):      number of registers. Must be in [2, 16].
                              Defaults to 4.
            data_width (int): bit width of each register. Defaults to 32.

        Raises:
            SimulationException: if depth is outside the range [2, 16].

        Example:
            >>> rf = RegisterFile(depth=8)
            >>> len(rf)
            8
        """
        if not (RF_MIN_DEPTH <= depth <= RF_MAX_DEPTH):
            raise SimulationException(
                f"Invalid RF depth: {depth}. "
                f"Must be between {RF_MIN_DEPTH} and {RF_MAX_DEPTH}."
            )
        self.depth: int = depth
        self.data_width: int = data_width
        self._registers: list[int] = [0] * depth

    def read(self, index: int) -> int:
        """
        Reads the value stored at the given register index.

        Args:
            index (int): register index (0-based).

        Returns:
            int: integer value stored in the register.

        Raises:
            SimulationException: if index is outside [0, depth-1].

        Example:
            >>> rf = RegisterFile(depth=4)
            >>> rf.write(2, 99)
            >>> rf.read(2)
            99
        """
        self._validate_index(index)
        return self._registers[index]

    def write(self, index: int, value: int) -> None:
        """
        Writes an integer value to the given register index.

        Args:
            index (int): register index (0-based).
            value (int): integer value to store.

        Raises:
            SimulationException: if index is outside [0, depth-1].

        Example:
            >>> rf = RegisterFile(depth=4)
            >>> rf.write(3, 7)
            >>> rf.read(3)
            7
        """
        self._validate_index(index)
        self._registers[index] = value

    def reset(self) -> None:
        """
        Sets all registers to 0.

        Used when resetting the simulation or initializing the PE.

        Example:
            >>> rf = RegisterFile(depth=4)
            >>> rf.write(0, 55)
            >>> rf.reset()
            >>> rf.read(0)
            0
        """
        self._registers = [0] * self.depth

    def get_state(self) -> list[int]:
        """
        Returns a copy of the current state of all registers.

        Returns a copy rather than a direct reference to prevent
        external code from modifying internal state without going
        through write().

        Returns:
            list[int]: list with the current values of all registers.

        Example:
            >>> rf = RegisterFile(depth=3)
            >>> rf.write(1, 5)
            >>> rf.get_state()
            [0, 5, 0]
        """
        return list(self._registers)

    def _validate_index(self, index: int) -> None:
        """
        Validates that the register index is within bounds.

        Args:
            index (int): index to validate.

        Raises:
            SimulationException: if index < 0 or index >= depth.
        """
        if not (0 <= index < self.depth):
            raise SimulationException(
                f"Register index out of range: {index}. "
                f"Valid range: [0, {self.depth - 1}]."
            )

    def __len__(self) -> int:
        """Returns the depth of the register file."""
        return self.depth