"""
PE Array module for the ArchPilot-CGRA simulator.

The PEArray is the top-level simulation engine. It instantiates and
manages the full mesh of Processing Elements, coordinates the Config
Memory for instruction dispatch, and delegates data routing to the NoC.

The simulation loop (step/run) follows this sequence each clock cycle:
  1. Distribute instructions from ConfigMemory to every PE.
  2. Propagate previous-cycle outputs through the NoC to PE neighbor inputs.
  3. Execute tick() on every PE simultaneously.
  4. Record activity trace for the current cycle.

Topology changes are forbidden during execution via the NoC lock
(SYRS-MOD-001). Multi-threaded reads on ConfigMemory are protected by
its internal lock (SYRS-REL-002).

Complies with: SYRS-FUN-001, SYRS-FUN-005, SYRS-FUN-006,
               SYRS-FUN-007, SYRS-FUN-008, SYRS-MOD-001,
               SYRS-REL-002, SYRS-MNT-001, SYRS-MNT-002,
               SYRS-QLY-002
"""

from typing import Any, Final

from archpilot_cgra.config_memory import ConfigMemory
from archpilot_cgra.exceptions import SimulationException
from archpilot_cgra.noc import SUPPORTED_TOPOLOGIES, TOPOLOGY_MESH, NoC
from archpilot_cgra.processing_element import ProcessingElement

# Default RF depth and data width for new PEArray instances
DEFAULT_RF_DEPTH: Final[int] = 4
DEFAULT_DATA_WIDTH: Final[int] = 32


class PEArray:
    """
    Bidimensional mesh of Processing Elements with cycle-accurate simulation.

    PEArray is the main entry point for running simulations. It creates
    the full NxM grid of PEs, manages the ConfigMemory for instruction
    distribution, delegates inter-PE data routing to the NoC, and
    accumulates an activity trace for PPA estimation.

    Complies with: SYRS-FUN-001 (parametric mesh creation),
                   SYRS-FUN-005 (cycle-based simulation engine),
                   SYRS-FUN-007 (activity trace recording),
                   SYRS-FUN-008 (ConfigMemory integration),
                   SYRS-FUN-006 (NoC topology support via set_topology).

    Attributes:
        rows (int):           number of mesh rows.
        cols (int):           number of mesh columns.
        current_cycle (int):  current clock cycle (incremented by step()).
        activity_trace (list): list of per-cycle activity snapshots.
        config_memory (ConfigMemory): centralized instruction store.
        noc (NoC):            data routing engine.

    Example:
        >>> array = PEArray(rows=4, cols=4)
        >>> array.rows, array.cols
        (4, 4)
        >>> len(array.get_all_pes())
        16
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        rf_depth: int = DEFAULT_RF_DEPTH,
        data_width: int = DEFAULT_DATA_WIDTH,
        topology: str = TOPOLOGY_MESH,
        config_memory: ConfigMemory | None = None,
    ) -> None:
        """
        Initialize the PE Array.

        Creates the rows×cols grid of PEs, the Config Memory and the NoC.
        All parameters can be specified independently per PE via the
        config_memory or by loading a configuration sequence later.

        When a ConfigMemory instance is injected, its dimensions must
        match rows and cols exactly. A mismatch would cause step() to
        crash with an unguarded KeyError because ConfigMemory.distribute()
        iterates its own rows/cols while PEArray._grid uses its own.

        Args:
            rows (int):          number of rows in the mesh (>= 1).
            cols (int):          number of columns in the mesh (>= 1).
            rf_depth (int):      Register File depth per PE (2–16).
                                 Defaults to 4.
            data_width (int):    data bit width per PE. Defaults to 32.
            topology (str):      NoC topology: "mesh" or "torus".
                                 Defaults to "mesh".
            config_memory:       pre-built ConfigMemory instance.
                                 If None, an empty one is created.
                                 If provided, its dimensions must equal
                                 rows × cols.

        Raises:
            SimulationException: if rows or cols < 1, topology is not
                                 supported, or an injected ConfigMemory
                                 has mismatched dimensions.

        Example:
            >>> array = PEArray(rows=2, cols=2, topology="torus")
            >>> array.noc.topology
            'torus'
        """
        if rows < 1 or cols < 1:
            raise SimulationException(
                f"Invalid mesh dimensions: {rows}x{cols}. Must be >= 1."
            )
        if topology not in SUPPORTED_TOPOLOGIES:
            raise SimulationException(
                f"Unsupported topology: '{topology}'. "
                f"Valid: {sorted(SUPPORTED_TOPOLOGIES)}."
            )

        # Validate injected ConfigMemory dimensions (prevents KeyError in step)
        if config_memory is not None:
            if config_memory.rows != rows or config_memory.cols != cols:
                raise SimulationException(
                    f"ConfigMemory dimensions ({config_memory.rows}x"
                    f"{config_memory.cols}) do not match PEArray dimensions "
                    f"({rows}x{cols})."
                )

        self.rows: int = rows
        self.cols: int = cols
        self.current_cycle: int = 0
        self.activity_trace: list[dict[str, Any]] = []

        # Build the PE grid (SYRS-FUN-001)
        self._grid: dict[tuple[int, int], ProcessingElement] = {
            (r, c): ProcessingElement(
                pe_id=(r, c),
                rf_depth=rf_depth,
                data_width=data_width,
            )
            for r in range(rows)
            for c in range(cols)
        }

        # Config Memory (SYRS-FUN-008)
        self.config_memory: ConfigMemory = config_memory or ConfigMemory(
            rows=rows, cols=cols
        )

        # NoC (SYRS-FUN-006)
        self.noc: NoC = NoC(rows=rows, cols=cols, topology=topology)

    # ------------------------------------------------------------------
    # Simulation engine (SYRS-FUN-005)
    # ------------------------------------------------------------------

    def step(self) -> dict[str, Any]:
        """
        Advance the simulation by exactly one clock cycle.

        Sequence:
          1. Lock topology (SYRS-MOD-001) on the first cycle.
          2. Distribute instructions from ConfigMemory to every PE.
          3. Propagate previous outputs through the NoC.
          4. Execute tick() on all PEs.
          5. Record activity trace (SYRS-FUN-007).
          6. Increment current_cycle.

        Returns:
            dict with keys:
              - "cycle" (int): the cycle that just completed.
              - "results" (dict[tuple, int]): pe_id -> output value.
              - "activity" (dict): per-PE activity snapshot.

        Raises:
            OverflowSimulationException: if any PE detects overflow.

        Example:
            >>> array = PEArray(rows=2, cols=2)
            >>> state = array.step()
            >>> "cycle" in state
            True
        """
        # Lock topology on first cycle (SYRS-MOD-001)
        if not self.noc.locked:
            self.noc.lock()

        # 1. Distribute instructions
        instructions = self.config_memory.distribute(self.current_cycle)
        for pe_id, instr in instructions.items():
            self._grid[pe_id].load_instruction(instr)

        # 2. Propagate outputs from previous cycle through the NoC
        self.noc.propagate(self._grid)

        # 3. Execute all PEs
        results: dict[tuple[int, int], int] = {}
        for pe_id, pe in self._grid.items():
            results[pe_id] = pe.tick()

        # 4. Record activity trace (SYRS-FUN-007)
        activity = self._collect_activity()
        self.activity_trace.append(activity)

        self.current_cycle += 1

        return {
            "cycle": self.current_cycle - 1,
            "results": results,
            "activity": activity,
        }

    def run(self, num_cycles: int) -> list[dict[str, Any]]:
        """
        Execute the simulation for a fixed number of clock cycles.

        Calls step() repeatedly and collects the results. The topology
        is locked on the first step() and remains locked until reset()
        is called.

        Args:
            num_cycles (int): number of cycles to simulate (>= 1).

        Returns:
            list of step() result dicts, one per cycle.

        Raises:
            SimulationException:          if num_cycles < 1.
            OverflowSimulationException:  if any PE overflows.

        Example:
            >>> array = PEArray(rows=2, cols=2)
            >>> history = array.run(5)
            >>> len(history)
            5
            >>> array.current_cycle
            5
        """
        if num_cycles < 1:
            raise SimulationException(
                f"num_cycles must be >= 1, got {num_cycles}."
            )
        return [self.step() for _ in range(num_cycles)]

    def reset(self) -> None:
        """
        Reset the entire mesh to its initial state.

        Resets every PE (registers, activity counter, output value),
        clears the activity trace, unlocks the NoC topology, and sets
        current_cycle back to 0. The ConfigMemory sequence is NOT
        cleared: the same program can be re-run after reset.

        Example:
            >>> array = PEArray(rows=2, cols=2)
            >>> array.run(3)
            [...]
            >>> array.reset()
            >>> array.current_cycle
            0
        """
        for pe in self._grid.values():
            pe.reset()
        self.current_cycle = 0
        self.activity_trace = []
        self.noc.unlock()

    # ------------------------------------------------------------------
    # Topology API (SYRS-FUN-006, ACT-12)
    # ------------------------------------------------------------------

    def set_topology(self, topology: str) -> None:
        """
        Change the NoC routing topology.

        Can only be called when the simulation is not running (i.e.,
        before the first step() or after reset()). Complies with
        SYRS-MOD-001.

        Args:
            topology (str): "mesh" or "torus".

        Raises:
            SimulationException: if the NoC is locked or topology is
                                 not supported.

        Example:
            >>> array = PEArray(rows=4, cols=4)
            >>> array.set_topology("torus")
            >>> array.noc.topology
            'torus'
        """
        self.noc.set_topology(topology)

    def get_topology(self) -> str:
        """
        Return the current NoC topology string.

        Returns:
            str: "mesh" or "torus".

        Example:
            >>> array = PEArray(rows=2, cols=2)
            >>> array.get_topology()
            'mesh'
        """
        return self.noc.topology

    # ------------------------------------------------------------------
    # PE access API
    # ------------------------------------------------------------------

    def get_pe(self, row: int, col: int) -> ProcessingElement:
        """
        Return the PE at the given mesh position.

        Args:
            row (int): row index.
            col (int): column index.

        Returns:
            ProcessingElement at (row, col).

        Raises:
            SimulationException: if (row, col) is outside the mesh.

        Example:
            >>> array = PEArray(rows=2, cols=2)
            >>> array.get_pe(0, 1).pe_id
            (0, 1)
        """
        if (row, col) not in self._grid:
            raise SimulationException(
                f"PE ({row}, {col}) is outside the {self.rows}x{self.cols} mesh."
            )
        return self._grid[(row, col)]

    def get_all_pes(self) -> list[ProcessingElement]:
        """
        Return all PEs ordered by (row, col).

        Returns:
            list of ProcessingElement in row-major order.

        Example:
            >>> array = PEArray(rows=2, cols=3)
            >>> len(array.get_all_pes())
            6
        """
        return [
            self._grid[(r, c)]
            for r in range(self.rows)
            for c in range(self.cols)
        ]

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def get_array_state(self) -> dict[str, Any]:
        """
        Return the complete mesh state for inspection or debugging.

        Returns:
            dict with keys:
              - "current_cycle" (int)
              - "topology" (str)
              - "pes" (dict[str, dict]): pe_id -> PE state snapshot.

        Example:
            >>> array = PEArray(rows=1, cols=1)
            >>> state = array.get_array_state()
            >>> state["current_cycle"]
            0
        """
        return {
            "current_cycle": self.current_cycle,
            "topology": self.noc.topology,
            "pes": {
                str(pe_id): pe.get_state()
                for pe_id, pe in self._grid.items()
            },
        }

    def get_activity_summary(self) -> dict[str, int]:
        """
        Return accumulated active cycles per PE.

        Useful for generating utilization heatmaps (SYRS-USA-004)
        and for feeding the PPA estimation engine.

        Returns:
            dict mapping str(pe_id) -> total active cycle count.

        Example:
            >>> array = PEArray(rows=2, cols=2)
            >>> array.run(3)
            [...]
            >>> summary = array.get_activity_summary()
            >>> isinstance(summary, dict)
            True
        """
        return {
            str(pe_id): pe.activity_count
            for pe_id, pe in self._grid.items()
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_activity(self) -> dict[str, Any]:
        """
        Snapshot the activity state of all PEs for the current cycle.

        Returns a dict mapping str(pe_id) to a dict with keys:
          - "active" (bool): whether the FU executed a real operation.
          - "opcode" (str): last opcode executed.
          - "output" (int): output_value of the PE this cycle.
        """
        return {
            str(pe_id): {
                "active": pe.fu.active,
                "opcode": pe.fu.last_opcode,
                "output": pe.output_value,
            }
            for pe_id, pe in self._grid.items()
        }