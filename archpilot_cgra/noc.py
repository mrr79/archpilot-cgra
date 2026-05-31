"""
Network-on-Chip (NoC) module for the ArchPilot-CGRA simulator.

The NoC manages data routing between Processing Elements in the mesh.
It supports two standard CGRA topologies: Mesh and Torus. In Mesh, edge
PEs have no neighbors beyond the boundary. In Torus, the mesh wraps
around so that every PE always has exactly four neighbors (north, south,
east, west), enabling uniform dataflow patterns.

The NoC operates in two phases per clock cycle:
  1. Snapshot: capture all current PE output values before any routing.
  2. Propagate: distribute captured values to the neighbor_inputs of
     every PE simultaneously, preventing order-of-update artifacts.

Topology changes are forbidden during execution (SYRS-MOD-001).

Complies with: SYRS-FUN-006, SYRS-MOD-001, SYRS-MNT-001,
               SYRS-MNT-002, SYRS-QLY-002
"""

from typing import TYPE_CHECKING, Final

from archpilot_cgra.exceptions import SimulationException

if TYPE_CHECKING:
    from archpilot_cgra.processing_element import ProcessingElement


# Supported topology identifiers
TOPOLOGY_MESH: Final[str] = "mesh"
TOPOLOGY_TORUS: Final[str] = "torus"
SUPPORTED_TOPOLOGIES: Final[frozenset[str]] = frozenset(
    {TOPOLOGY_MESH, TOPOLOGY_TORUS}
)

# Cardinal direction labels used in ProcessingElement.neighbor_inputs.
# All identifiers use English to match the rest of the public API.
DIRECTION_NORTH: Final[str] = "north"
DIRECTION_SOUTH: Final[str] = "south"
DIRECTION_EAST: Final[str] = "east"
DIRECTION_WEST: Final[str] = "west"


class NoC:
    """
    Network-on-Chip for a CGRA PE mesh.

    Implements data routing between PEs using cardinal-direction
    connections (north, south, east, west). Supports Mesh and Torus
    topologies (SYRS-FUN-006). The topology cannot be changed while
    the simulation is running (SYRS-MOD-001).

    In Mesh topology:
        - Edge PEs have no neighbor beyond the boundary.
        - Boundary neighbor inputs remain at 0.

    In Torus topology:
        - The mesh wraps around in both dimensions.
        - Every PE always has exactly four active neighbors.

    Attributes:
        rows (int):      number of mesh rows.
        cols (int):      number of mesh columns.
        topology (str):  active topology ("mesh" or "torus").
        locked (bool):   True when topology changes are forbidden
                         (i.e., simulation is running).

    Example:
        >>> noc = NoC(rows=2, cols=2, topology="mesh")
        >>> noc.topology
        'mesh'
        >>> noc.locked
        False
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        topology: str = TOPOLOGY_MESH,
    ) -> None:
        """
        Initialize the NoC.

        Args:
            rows (int):     number of mesh rows (>= 1).
            cols (int):     number of mesh columns (>= 1).
            topology (str): "mesh" or "torus". Defaults to "mesh".

        Raises:
            SimulationException: if rows or cols < 1, or topology is
                                 not supported.

        Example:
            >>> noc = NoC(rows=4, cols=4, topology="torus")
            >>> noc.topology
            'torus'
        """
        if rows < 1 or cols < 1:
            raise SimulationException(
                f"Invalid NoC dimensions: {rows}x{cols}. Must be >= 1."
            )
        if topology not in SUPPORTED_TOPOLOGIES:
            raise SimulationException(
                f"Unsupported topology: '{topology}'. "
                f"Valid options: {sorted(SUPPORTED_TOPOLOGIES)}."
            )
        self.rows: int = rows
        self.cols: int = cols
        self.topology: str = topology
        self.locked: bool = False

    # ------------------------------------------------------------------
    # Topology lock (SYRS-MOD-001)
    # ------------------------------------------------------------------

    def lock(self) -> None:
        """
        Lock the topology to prevent changes during execution.

        Called by PEArray.step() before the first cycle and by any
        method that starts a simulation run. Once locked, set_topology()
        raises SimulationException.

        Complies with SYRS-MOD-001: topology changes are forbidden
        while the simulation is running.

        Example:
            >>> noc = NoC(rows=2, cols=2)
            >>> noc.lock()
            >>> noc.locked
            True
        """
        self.locked = True

    def unlock(self) -> None:
        """
        Unlock the topology to allow reconfiguration.

        Called by PEArray.reset() to restore editability after
        a simulation run completes or is interrupted.

        Example:
            >>> noc = NoC(rows=2, cols=2)
            >>> noc.lock()
            >>> noc.unlock()
            >>> noc.locked
            False
        """
        self.locked = False

    def set_topology(self, topology: str) -> None:
        """
        Change the active topology.

        Cannot be called while the NoC is locked (i.e., simulation is
        running). Complies with SYRS-MOD-001.

        Args:
            topology (str): "mesh" or "torus".

        Raises:
            SimulationException: if the NoC is locked or if topology
                                 is not a supported value.

        Example:
            >>> noc = NoC(rows=2, cols=2, topology="mesh")
            >>> noc.set_topology("torus")
            >>> noc.topology
            'torus'
        """
        if self.locked:
            raise SimulationException(
                "Cannot change NoC topology while the simulation is running. "
                "Call reset() first to unlock the topology."
            )
        if topology not in SUPPORTED_TOPOLOGIES:
            raise SimulationException(
                f"Unsupported topology: '{topology}'. "
                f"Valid options: {sorted(SUPPORTED_TOPOLOGIES)}."
            )
        self.topology = topology

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def propagate(
        self,
        grid: dict[tuple[int, int], "ProcessingElement"],
    ) -> None:
        """
        Propagate PE output values to neighbor inputs for one cycle.

        Implements a two-phase snapshot-then-distribute strategy to
        ensure that all PEs read the values produced in the *previous*
        cycle, not values being written by peers in the current cycle.

        Phase 1 — Snapshot: read output_value from every PE.
        Phase 2 — Distribute: write neighbor inputs via get_neighbors(),
                  which delegates to the correct topology internally.

        This method is called by PEArray.step() before each tick() in
        the simulation loop.

        Args:
            grid: dict mapping pe_id (row, col) -> ProcessingElement.

        Example:
            >>> # After pe.tick(), noc.propagate(grid) routes outputs.
            >>> noc = NoC(rows=2, cols=1, topology="mesh")
        """
        # Phase 1: snapshot all outputs atomically
        snapshot: dict[tuple[int, int], int] = {
            pe_id: pe.output_value for pe_id, pe in grid.items()
        }

        # Phase 2: distribute to neighbor inputs.
        # get_neighbors() already dispatches to the correct topology,
        # so a single loop replaces the former _propagate_mesh /
        # _propagate_torus duplication.
        for (r, c), pe in grid.items():
            neighbors = self.get_neighbors(r, c)
            self._apply_neighbor_inputs(pe, neighbors, snapshot)

    def get_neighbors(
        self, row: int, col: int
    ) -> dict[str, tuple[int, int] | None]:
        """
        Return the neighbor coordinates for a given PE position.

        In Mesh topology, boundary neighbors are None (no connection).
        In Torus topology, all neighbors wrap around (never None).

        Args:
            row (int): PE row index.
            col (int): PE column index.

        Returns:
            dict mapping direction string to (row, col) tuple or None.
            Keys: "north", "south", "east", "west".

        Example:
            >>> noc = NoC(rows=3, cols=3, topology="mesh")
            >>> noc.get_neighbors(0, 0)["north"] is None
            True
            >>> noc = NoC(rows=3, cols=3, topology="torus")
            >>> noc.get_neighbors(0, 0)["north"]
            (2, 0)
        """
        if self.topology == TOPOLOGY_MESH:
            return self._neighbors_mesh(row, col)
        return self._neighbors_torus(row, col)

    # ------------------------------------------------------------------
    # Private routing implementations
    # ------------------------------------------------------------------

    def _neighbors_mesh(
        self, row: int, col: int
    ) -> dict[str, tuple[int, int] | None]:
        """
        Compute Mesh neighbors; returns None for out-of-bounds positions.

        A PE at (row, col) has neighbors:
          - north: (row-1, col) if row > 0,          else None
          - south: (row+1, col) if row < rows-1,      else None
          - east:  (row, col+1) if col < cols-1,      else None
          - west:  (row, col-1) if col > 0,           else None
        """
        return {
            DIRECTION_NORTH: (row - 1, col) if row > 0 else None,
            DIRECTION_SOUTH: (row + 1, col) if row < self.rows - 1 else None,
            DIRECTION_EAST:  (row, col + 1) if col < self.cols - 1 else None,
            DIRECTION_WEST:  (row, col - 1) if col > 0 else None,
        }

    def _neighbors_torus(
        self, row: int, col: int
    ) -> dict[str, tuple[int, int]]:
        """
        Compute Torus neighbors using modular arithmetic.

        Every PE always has exactly four neighbors; the return type
        never includes None because wrap-around connections always exist.
        """
        return {
            DIRECTION_NORTH: ((row - 1) % self.rows, col),
            DIRECTION_SOUTH: ((row + 1) % self.rows, col),
            DIRECTION_EAST:  (row, (col + 1) % self.cols),
            DIRECTION_WEST:  (row, (col - 1) % self.cols),
        }

    @staticmethod
    def _apply_neighbor_inputs(
        pe: "ProcessingElement",
        neighbors: dict[str, tuple[int, int] | None],
        snapshot: dict[tuple[int, int], int],
    ) -> None:
        """
        Write snapshot values into PE neighbor_inputs.

        For each direction:
          - If the neighbor exists in the snapshot, write its value.
          - If the neighbor is None (Mesh boundary), write 0.
        """
        for direction, neighbor_id in neighbors.items():
            value = snapshot[neighbor_id] if neighbor_id is not None else 0
            pe.neighbor_inputs[direction] = value