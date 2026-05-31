"""
Processing Element (PE) module for the CGRA architecture.

The PE is the fundamental compute unit of the mesh. It encapsulates
a Functional Unit (FU), a Register File (RF), and input multiplexer
logic for communicating with neighbors through the NoC.

Complies with: SYRS-FUN-002, SYRS-FUN-003, SYRS-FUN-004, SYRS-FUN-010,
               SYRS-MNT-001, SYRS-MNT-002, SYRS-QLY-002
"""

from typing import Any, Final, TypedDict

from archpilot_cgra.exceptions import (
    OverflowSimulationException,
    SimulationException,
)
from archpilot_cgra.functional_unit import FunctionalUnit
from archpilot_cgra.register_file import RegisterFile


class Instruction(TypedDict):
    """
    Typed dictionary representing a PE configuration instruction.

    Using TypedDict gives type checkers (mypy/pyright) the ability to
    catch missing keys or wrong value types at analysis time, with zero
    runtime overhead.

    Fields:
        opcode (str):  operation to execute.
        src_a (int):   RF index for operand A (used when mux_sel == 0).
        src_b (int):   RF index for operand B.
        dst (int):     RF index where the result is written.
        mux_sel (int): input mux selector (0=RF, 1=north, 2=south,
                       3=east, 4=west).
    """

    opcode: str
    src_a: int
    src_b: int
    dst: int
    mux_sel: int


# Valid neighbor directions for NoC input.
# All labels use English to match the rest of the public API.
# Final prevents accidental reassignment of this module-level constant.
VALID_DIRECTIONS: Final[frozenset[str]] = frozenset(
    {"north", "south", "east", "west"}
)

# Maps mux_sel integer values to neighbor direction strings.
# Final prevents accidental reassignment of this module-level constant.
MUX_DIRECTION_MAP: Final[dict[int, str]] = {
    1: "north",
    2: "south",
    3: "east",
    4: "west",
}


class ProcessingElement:
    """
    Processing Element (PE) of the CGRA architecture.

    Integrates a Functional Unit (FU), a Register File (RF), and a
    configurable input multiplexer, allowing each PE to operate
    autonomously each clock cycle based on an instruction distributed
    by the Config Memory.

    Each cycle the PE:
      1. Reads operands from the RF or from neighbor inputs (NoC).
      2. Executes the operation indicated by the current instruction.
      3. Checks for overflow and raises an exception if needed.
      4. Writes the result to the destination register in the RF.
      5. Updates the activity trace counter.

    __slots__ is declared to eliminate the per-instance __dict__,
    saving ~200-400 bytes per object. In a 16×16 mesh with multiple
    sub-objects this removes 768+ extra dicts with zero behavioral change.

    Complies with: SYRS-FUN-002, SYRS-FUN-010.

    Attributes:
        pe_id (tuple[int, int]):   identifier (row, column) in the mesh.
        fu (FunctionalUnit):       functional unit of the PE.
        rf (RegisterFile):         register file of the PE.
        data_width (int):          bit width of the data.
        output_value (int):        last result produced by the PE.
        input_mux_sel (int):       input mux selector (0-4).
        activity_count (int):      accumulated active cycles (SYRS-FUN-007).
        neighbor_inputs (dict):    cardinal neighbor input values.

    Example:
        >>> pe = ProcessingElement(pe_id=(0, 0), rf_depth=4, data_width=32)
        >>> pe.rf.write(0, 10)
        >>> pe.rf.write(1, 5)
        >>> pe.load_instruction(
        ...     Instruction(opcode="ADD", src_a=0, src_b=1, dst=2, mux_sel=0)
        ... )
        >>> pe.tick()
        15
        >>> pe.rf.read(2)
        15
    """

    __slots__ = (
        "pe_id",
        "data_width",
        "fu",
        "rf",
        "input_mux_sel",
        "output_value",
        "activity_count",
        "neighbor_inputs",
        "_current_instruction",
    )

    def __init__(
        self,
        pe_id: tuple[int, int],
        rf_depth: int = 4,
        data_width: int = 32,
    ) -> None:
        """
        Initializes the Processing Element.

        Args:
            pe_id (tuple[int, int]): identifier (row, column) in the mesh.
            rf_depth (int):          RF depth, between 2 and 16.
                                     Defaults to 4.
            data_width (int):        bit width for operands and results.
                                     Defaults to 32 bits.

        Example:
            >>> pe = ProcessingElement(pe_id=(1, 2), rf_depth=8)
            >>> pe.pe_id
            (1, 2)
            >>> len(pe.rf)
            8
        """
        self.pe_id: tuple[int, int] = pe_id
        self.data_width: int = data_width
        self.fu: FunctionalUnit = FunctionalUnit(data_width=data_width)
        self.rf: RegisterFile = RegisterFile(depth=rf_depth, data_width=data_width)
        self.input_mux_sel: int = 0
        self.output_value: int = 0
        self.activity_count: int = 0
        self.neighbor_inputs: dict[str, int] = {
            "north": 0,
            "south": 0,
            "east": 0,
            "west": 0,
        }
        # X | None is the modern Python 3.10+ union syntax.
        # Replaces the deprecated Optional[X] from typing.
        self._current_instruction: Instruction | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_instruction(self, instruction: Instruction) -> None:
        """
        Loads the configuration instruction for the next cycle.

        Using Instruction (TypedDict) instead of a plain dict allows
        type checkers to catch missing keys or wrong value types in
        callers at analysis time.

        Args:
            instruction (Instruction): typed instruction with opcode,
                src_a, src_b, dst and mux_sel fields.

        Example:
            >>> pe = ProcessingElement((0, 0))
            >>> pe.load_instruction(
            ...     Instruction(opcode="MUL", src_a=0, src_b=1,
            ...                 dst=2, mux_sel=0)
            ... )
        """
        self._current_instruction = instruction
        self.input_mux_sel = instruction.get("mux_sel", 0)

    def tick(self) -> int:
        """
        Advances the simulation by exactly one clock cycle.

        Execution sequence:
          1. Read operands according to the instruction and mux selector.
          2. Execute the FU with the obtained operands.
          3. Check for overflow (SYRS-FUN-010).
          4. Write the result to the destination register.
          5. Update the activity trace counter (SYRS-FUN-007).

        Direct key indexing (instr["key"]) is used instead of
        instr.get("key", default) because Instruction is a TypedDict
        with required keys — all keys are guaranteed to exist at static
        analysis time. Using .get() with defaults would silently swallow
        missing-key errors, defeating the purpose of TypedDict.

        Returns:
            int: value produced in this cycle (stored in output_value).

        Raises:
            OverflowSimulationException: if the result is outside the
                signed N-bit range [-(2^(N-1)), 2^(N-1)-1].

        Example:
            >>> pe = ProcessingElement((0, 0), rf_depth=4, data_width=32)
            >>> pe.rf.write(0, 6)
            >>> pe.rf.write(1, 7)
            >>> pe.load_instruction(
            ...     Instruction(opcode="MUL", src_a=0, src_b=1,
            ...                 dst=3, mux_sel=0)
            ... )
            >>> pe.tick()
            42
        """
        if self._current_instruction is None:
            return 0

        instr = self._current_instruction
        # Direct indexing: TypedDict guarantees these keys always exist.
        opcode: str    = instr["opcode"]
        src_a_idx: int = instr["src_a"]
        src_b_idx: int = instr["src_b"]
        dst_idx: int   = instr["dst"]

        operand_a: int = self._read_mux(src_a_idx)
        operand_b: int = self.rf.read(src_b_idx)
        result: int = self.fu.execute(opcode, operand_a, operand_b)

        self._check_overflow(result)

        self.rf.write(dst_idx, result)
        self.output_value = result

        if self.fu.active:
            self.activity_count += 1

        return result

    def set_neighbor_input(self, direction: str, value: int) -> None:
        """
        Sets the input value received from a neighbor via the NoC.

        Args:
            direction (str): cardinal direction of the neighbor:
                             "north", "south", "east", "west".
            value (int):     data value received from the neighbor.

        Raises:
            SimulationException: if direction is not a valid cardinal.

        Example:
            >>> pe = ProcessingElement((1, 1))
            >>> pe.set_neighbor_input("north", 99)
            >>> pe.neighbor_inputs["north"]
            99
        """
        if direction not in VALID_DIRECTIONS:
            raise SimulationException(
                f"Invalid direction: '{direction}'. "
                f"Valid directions: {sorted(VALID_DIRECTIONS)}."
            )
        self.neighbor_inputs[direction] = value

    def get_state(self) -> dict[str, Any]:
        """
        Returns the complete PE state for inspection or debugging.

        Useful for interactive debug mode (SYRS-OPS-002) and for
        building the activity trace (SYRS-FUN-007).

        Returns:
            dict[str, Any]: with keys: pe_id, registers, output_value,
                last_opcode, activity_count, input_mux_sel,
                neighbor_inputs.

        Example:
            >>> pe = ProcessingElement((0, 0))
            >>> state = pe.get_state()
            >>> "pe_id" in state
            True
        """
        return {
            "pe_id": self.pe_id,
            "registers": self.rf.get_state(),
            "output_value": self.output_value,
            "last_opcode": self.fu.last_opcode,
            "activity_count": self.activity_count,
            "input_mux_sel": self.input_mux_sel,
            "neighbor_inputs": dict(self.neighbor_inputs),
        }

    def reset(self) -> None:
        """
        Resets the PE to its initial simulation state.

        Clears the RF, resets the FU, sets output_value and
        activity_count to 0, and discards the current instruction.

        Example:
            >>> pe = ProcessingElement((0, 0))
            >>> pe.rf.write(0, 99)
            >>> pe.reset()
            >>> pe.rf.read(0)
            0
        """
        self.rf.reset()
        self.fu.reset()
        self.output_value = 0
        self.activity_count = 0
        self._current_instruction = None
        self.neighbor_inputs = {"north": 0, "south": 0, "east": 0, "west": 0}

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _read_mux(self, src_a_idx: int) -> int:
        """
        Reads operand A according to the input multiplexer selector.

        If mux_sel is 0, reads from the local RF at src_a_idx.
        If mux_sel is 1-4, reads from the corresponding neighbor input.
        If mux_sel is any other value, raises SimulationException to
        surface configuration errors rather than silently falling back
        to the RF (which would hide misconfigured instructions).

        Args:
            src_a_idx (int): RF index to use when mux_sel == 0.

        Returns:
            int: value of operand A.

        Raises:
            SimulationException: if mux_sel is outside the valid range [0, 4].
        """
        if self.input_mux_sel == 0:
            return self.rf.read(src_a_idx)
        direction = MUX_DIRECTION_MAP.get(self.input_mux_sel)
        if direction is None:
            # Raise explicitly rather than silently falling back to RF.
            # A misconfigured instruction must produce an error, not a
            # wrong result with no signal.
            raise SimulationException(
                f"Invalid mux_sel: {self.input_mux_sel}. "
                f"Valid range: [0, 4]."
            )
        return self.neighbor_inputs[direction]

    def _check_overflow(self, result: int) -> None:
        """
        Verifies that the result is within the signed N-bit range.

        The signed N-bit range is [-(2^(N-1)), 2^(N-1)-1].
        The bounds are NOT symmetric: for 8-bit, the range is [-128, 127].

        The previous implementation used abs(result) > max_val, which
        incorrectly flagged -128 as overflow because abs(-128) = 128 > 127,
        even though -128 is the valid minimum for 8-bit signed integers.
        The correct check is: not (min_val <= result <= max_val).

        Args:
            result (int): value to verify.

        Raises:
            OverflowSimulationException: if result is outside
                [min_val, max_val].
        """
        max_val: int = (2 ** (self.data_width - 1)) - 1
        min_val: int = -(2 ** (self.data_width - 1))
        if not (min_val <= result <= max_val):
            raise OverflowSimulationException(
                pe_id=self.pe_id,
                value=result,
                max_value=max_val,
                pe_state=self.get_state(),
            )