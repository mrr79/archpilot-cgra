"""
Functional Unit (FU) module for the Processing Element.

The Functional Unit is the main arithmetic-logic component of each PE.
It supports a set of integer operations and maintains activity state
for switching trace generation.

Complies with: SYRS-FUN-003, SYRS-MNT-001, SYRS-MNT-002, SYRS-QLY-002
"""

from typing import Final

from archpilot_cgra.exceptions import InvalidOperationException

# Valid opcodes supported by the FU.
# Final prevents accidental reassignment of this module-level constant.
SUPPORTED_OPCODES: Final[frozenset[str]] = frozenset(
    {"ADD", "COMPLEMENT", "MUL", "NOP"}
)


class FunctionalUnit:
    """
    Functional Unit (FU) of a CGRA Processing Element.

    Executes integer arithmetic operations per clock cycle. Supports
    addition (ADD), two's complement (COMPLEMENT), multiplication (MUL),
    and no-operation (NOP), complying with SYRS-FUN-003.

    Maintains the last executed opcode and an activity flag to enable
    switching trace generation (SYRS-FUN-007).

    Attributes:
        data_width (int): bit width of the processed data.
        last_opcode (str): last opcode executed by the FU.
        active (bool): True if the FU executed a real operation
                       (other than NOP) in the last cycle.

    Example:
        >>> fu = FunctionalUnit(data_width=32)
        >>> fu.execute("ADD", 10, 5)
        15
        >>> fu.execute("COMPLEMENT", 7)
        -7
        >>> fu.execute("MUL", 3, 4)
        12
        >>> fu.execute("NOP", 0)
        0
    """

    def __init__(self, data_width: int = 32) -> None:
        """
        Initializes the Functional Unit.

        Args:
            data_width (int): bit width of the operands and result.
                              Defaults to 32 bits.

        Example:
            >>> fu = FunctionalUnit(data_width=16)
            >>> fu.data_width
            16
        """
        self.data_width: int = data_width
        self.last_opcode: str = "NOP"
        self.active: bool = False

    def execute(
        self,
        opcode: str,
        operand_a: int,
        operand_b: int = 0,
    ) -> int:
        """
        Executes an arithmetic operation on the given operands.

        Supported operations (SYRS-FUN-003):
          - ``ADD``:        result = operand_a + operand_b
          - ``COMPLEMENT``: result = -operand_a (two's complement)
          - ``MUL``:        result = operand_a * operand_b
          - ``NOP``:        result = 0, no activity recorded

        Uses Python 3.10+ match statement as the idiomatic dispatch
        construct for opcode handling.

        Args:
            opcode (str):    operation code to execute.
            operand_a (int): first integer operand.
            operand_b (int): second integer operand.
                             Ignored in COMPLEMENT and NOP. Defaults to 0.

        Returns:
            int: integer result of the operation.

        Raises:
            InvalidOperationException: if the opcode is not in
                SUPPORTED_OPCODES.

        Example:
            >>> fu = FunctionalUnit()
            >>> fu.execute("ADD", 3, 4)
            7
            >>> fu.execute("MUL", 6, 7)
            42
            >>> fu.execute("COMPLEMENT", 5)
            -5
        """
        if opcode not in SUPPORTED_OPCODES:
            raise InvalidOperationException(opcode)

        self.last_opcode = opcode
        self.active = opcode != "NOP"

        # match is the idiomatic Python 3.10+ construct for opcode dispatch.
        # It is more readable than chained if-elif and allows exhaustiveness
        # checking by type checkers.
        match opcode:
            case "ADD":
                return operand_a + operand_b
            case "COMPLEMENT":
                return -operand_a
            case "MUL":
                return operand_a * operand_b
            case _:  # NOP
                return 0

    def reset(self) -> None:
        """
        Resets the internal state of the FU to its initial values.

        Sets last_opcode to "NOP" and active to False. Called at the
        start of each new simulation context.

        Example:
            >>> fu = FunctionalUnit()
            >>> fu.execute("ADD", 1, 1)
            2
            >>> fu.reset()
            >>> fu.last_opcode
            'NOP'
            >>> fu.active
            False
        """
        self.last_opcode = "NOP"
        self.active = False
