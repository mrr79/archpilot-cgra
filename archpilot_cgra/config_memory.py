"""
Config Memory module for the ArchPilot-CGRA simulator.

The Config Memory is the centralized configuration store that distributes
control words (instructions) to every PE in the mesh on each clock cycle.
It is the mechanism that implements dynamic reconfiguration at cycle level,
which is the defining characteristic of CGRA architectures.

All read/write operations are protected against race conditions in
multi-threaded execution environments.

Complies with: SYRS-FUN-008, SYRS-REL-002, SYRS-MNT-001,
               SYRS-MNT-002, SYRS-QLY-002
"""

import threading
from typing import Any, Final

from archpilot_cgra.exceptions import ConfigMemoryException, SimulationException


# Default NOP instruction returned when no configuration is defined
# for a given PE or cycle.
NOP_INSTRUCTION: Final[dict[str, Any]] = {
    "opcode": "NOP",
    "src_a": 0,
    "src_b": 0,
    "dst": 0,
    "mux_sel": 0,
}


class ConfigMemory:
    """
    Centralized Configuration Memory for the CGRA mesh.

    Stores a sequence of per-cycle configuration words (instructions)
    for every PE in the mesh. On each clock cycle, the motor calls
    distribute() to obtain all instructions for that cycle and load
    them into the corresponding PEs.

    Read and write operations on the internal sequence are protected by
    a threading.Lock, preventing race conditions when the simulation runs
    with multiple threads (SYRS-REL-002). This includes the num_cycles
    property, which acquires the lock before reading len(_sequence).

    If no instruction is defined for a given (cycle, pe_id) pair, a NOP
    instruction is returned, ensuring safe fallback behavior.

    Attributes:
        rows (int):      number of rows in the mesh.
        cols (int):      number of columns in the mesh.
        num_cycles (int): number of configured cycles currently stored.

    Example:
        >>> seq = [
        ...     {(0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1,
        ...               "dst": 2, "mux_sel": 0}}
        ... ]
        >>> mem = ConfigMemory(rows=1, cols=1, config_sequence=seq)
        >>> mem.get_instruction(cycle=0, pe_id=(0, 0))["opcode"]
        'ADD'
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        config_sequence: list[dict[tuple[int, int], dict[str, Any]]] | None = None,
    ) -> None:
        """
        Initialize the Config Memory.

        Args:
            rows (int):            number of mesh rows (>= 1).
            cols (int):            number of mesh columns (>= 1).
            config_sequence:       optional initial configuration sequence.
                                   Each element is a dict mapping pe_id
                                   (row, col) to an instruction dict.
                                   If None, the memory starts empty (all NOPs).

        Raises:
            SimulationException:  if rows or cols are less than 1.
            ConfigMemoryException: if config_sequence contains a pe_id
                                   outside the mesh bounds.

        Example:
            >>> mem = ConfigMemory(rows=2, cols=2)
            >>> mem.num_cycles
            0
        """
        if rows < 1 or cols < 1:
            raise SimulationException(
                f"Invalid mesh dimensions: {rows}x{cols}. Must be >= 1."
            )
        self.rows: int = rows
        self.cols: int = cols
        self._sequence: list[dict[tuple[int, int], dict[str, Any]]] = []
        self._lock: threading.Lock = threading.Lock()

        if config_sequence is not None:
            self.load_sequence(config_sequence)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_sequence(
        self,
        config_sequence: list[dict[tuple[int, int], dict[str, Any]]],
    ) -> None:
        """
        Load a new configuration sequence into the memory.

        Replaces the entire current sequence. The operation is atomic:
        the internal state is only updated after full validation.

        Args:
            config_sequence: list where each element is a dict mapping
                             pe_id (tuple[int, int]) -> instruction dict.
                             Cycles are 0-indexed by position in the list.

        Raises:
            ConfigMemoryException: if any pe_id in the sequence refers to
                                   a PE outside the mesh bounds.

        Example:
            >>> mem = ConfigMemory(rows=2, cols=2)
            >>> seq = [{(0, 0): {"opcode": "NOP", "src_a": 0,
            ...                   "src_b": 0, "dst": 0, "mux_sel": 0}}]
            >>> mem.load_sequence(seq)
            >>> mem.num_cycles
            1
        """
        self._validate_sequence(config_sequence)
        with self._lock:
            self._sequence = [dict(cycle_cfg) for cycle_cfg in config_sequence]

    def get_instruction(
        self,
        cycle: int,
        pe_id: tuple[int, int],
    ) -> dict[str, Any]:
        """
        Return the instruction for a specific PE on a specific cycle.

        If no instruction is defined for the (cycle, pe_id) pair —
        either because the cycle is beyond the sequence length or because
        the PE was not configured for that cycle — a NOP instruction is
        returned.

        This method is thread-safe (SYRS-REL-002).

        Args:
            cycle (int):             clock cycle index (0-based).
            pe_id (tuple[int, int]): PE identifier (row, col).

        Returns:
            dict[str, Any]: instruction dict with keys opcode, src_a,
                            src_b, dst, mux_sel.

        Example:
            >>> mem = ConfigMemory(rows=1, cols=1)
            >>> mem.get_instruction(0, (0, 0))["opcode"]
            'NOP'
        """
        with self._lock:
            if cycle >= len(self._sequence):
                return dict(NOP_INSTRUCTION)
            return dict(self._sequence[cycle].get(pe_id, NOP_INSTRUCTION))

    def distribute(self, cycle: int) -> dict[tuple[int, int], dict[str, Any]]:
        """
        Distribute all instructions for a given cycle across the full mesh.

        Returns one instruction per PE. PEs not explicitly configured for
        the cycle receive a NOP instruction.

        This method is the primary interface used by the simulation engine
        (PEArray.step()) to load instructions into PEs each clock cycle.
        It is thread-safe (SYRS-REL-002).

        Args:
            cycle (int): clock cycle index (0-based).

        Returns:
            dict[tuple[int, int], dict[str, Any]]: mapping from pe_id
                to its instruction for this cycle. Contains exactly
                rows × cols entries.

        Example:
            >>> mem = ConfigMemory(rows=2, cols=2)
            >>> result = mem.distribute(0)
            >>> len(result)
            4
            >>> result[(0, 0)]["opcode"]
            'NOP'
        """
        with self._lock:
            cycle_cfg: dict[tuple[int, int], dict[str, Any]] = (
                self._sequence[cycle] if cycle < len(self._sequence) else {}
            )
            return {
                (r, c): dict(cycle_cfg.get((r, c), NOP_INSTRUCTION))
                for r in range(self.rows)
                for c in range(self.cols)
            }

    def append_cycle(
        self,
        cycle_config: dict[tuple[int, int], dict[str, Any]],
    ) -> None:
        """
        Append a single cycle configuration to the end of the sequence.

        Useful for building configurations incrementally without
        reloading the full sequence.

        Args:
            cycle_config: dict mapping pe_id -> instruction for the new cycle.

        Raises:
            ConfigMemoryException: if any pe_id is outside the mesh bounds.

        Example:
            >>> mem = ConfigMemory(rows=1, cols=1)
            >>> mem.append_cycle({(0, 0): {"opcode": "ADD", "src_a": 0,
            ...                            "src_b": 1, "dst": 2, "mux_sel": 0}})
            >>> mem.num_cycles
            1
        """
        self._validate_sequence([cycle_config])
        with self._lock:
            self._sequence.append(dict(cycle_config))

    def clear(self) -> None:
        """
        Remove all stored cycle configurations.

        After calling this method, all PEs will receive NOP instructions
        on every cycle until a new sequence is loaded.

        Example:
            >>> mem = ConfigMemory(rows=1, cols=1)
            >>> mem.append_cycle({})
            >>> mem.clear()
            >>> mem.num_cycles
            0
        """
        with self._lock:
            self._sequence = []

    @property
    def num_cycles(self) -> int:
        """
        Number of configured cycles currently stored.

        Acquires the internal lock before reading _sequence so that
        concurrent writes (load_sequence, append_cycle, clear) cannot
        produce a torn read (SYRS-REL-002).

        Example:
            >>> mem = ConfigMemory(rows=1, cols=1)
            >>> mem.num_cycles
            0
        """
        with self._lock:
            return len(self._sequence)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _validate_sequence(
        self,
        config_sequence: list[dict[tuple[int, int], dict[str, Any]]],
    ) -> None:
        """
        Validate that all pe_ids in a sequence are within mesh bounds.

        Args:
            config_sequence: sequence to validate.

        Raises:
            ConfigMemoryException: if any pe_id is out of bounds.
        """
        for cycle_idx, cycle_cfg in enumerate(config_sequence):
            for pe_id in cycle_cfg:
                r, c = pe_id
                if not (0 <= r < self.rows and 0 <= c < self.cols):
                    raise ConfigMemoryException(
                        f"Cycle {cycle_idx}: pe_id {pe_id} is outside "
                        f"the {self.rows}x{self.cols} mesh."
                    )