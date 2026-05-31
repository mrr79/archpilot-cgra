"""
Integration test: full system flow.

Exercises the complete simulation pipeline end-to-end:

    Python API → ConfigMemory → PEArray → NoC → PE (FU + RF) → result

Four scenarios:

1. Parallel independent computation (no inter-PE communication).
2. Dataflow pipeline via NoC Mesh (west link).
3. Dataflow pipeline via NoC Torus (wrap-around east link).
4. Multi-cycle accumulation stress test.

Direction labels use English ("north", "south", "east", "west") to
match the standardized public API (PR review: Must Fix — direction names).

Complies with: SYRS-FUN-001, SYRS-FUN-005, SYRS-FUN-006,
               SYRS-FUN-007, SYRS-FUN-008, SYRS-REL-001
"""

from archpilot_cgra.config_memory import ConfigMemory
from archpilot_cgra.noc import TOPOLOGY_TORUS
from archpilot_cgra.pe_array import PEArray


def make_instr(
    opcode: str,
    src_a: int = 0,
    src_b: int = 1,
    dst: int = 0,
    mux_sel: int = 0,
) -> dict:
    """Build an instruction dict."""
    return {
        "opcode": opcode,
        "src_a": src_a,
        "src_b": src_b,
        "dst": dst,
        "mux_sel": mux_sel,
    }


NOP = make_instr("NOP", 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Scenario 1: Parallel independent computation
# ---------------------------------------------------------------------------

class TestFullFlowParallelComputation:
    """
    Four PEs in a 2x2 mesh (rf_depth=8) execute independent sequences.
    No inter-PE communication (mux_sel=0 throughout).

    Reference model:
      PE(0,0): ADD(3,4)=7   -> MUL(7,2)=14   -> rf[0]=14
      PE(0,1): MUL(5,6)=30  -> ADD(30,10)=40  -> rf[0]=40
      PE(1,0): COMPL(9)=-9  -> ADD(-9,1)=-8   -> rf[0]=-8
      PE(1,1): ADD(1,1)=2   -> ADD(2,2)=4     -> rf[0]=4
    """

    def setup_method(self) -> None:
        c0 = {
            (0, 0): make_instr("ADD",        src_a=0, src_b=1, dst=0),
            (0, 1): make_instr("MUL",        src_a=0, src_b=1, dst=0),
            (1, 0): make_instr("COMPLEMENT", src_a=0, src_b=0, dst=0),
            (1, 1): make_instr("ADD",        src_a=0, src_b=1, dst=0),
        }
        c1 = {
            (0, 0): make_instr("MUL", src_a=0, src_b=2, dst=0),
            (0, 1): make_instr("ADD", src_a=0, src_b=3, dst=0),
            (1, 0): make_instr("ADD", src_a=0, src_b=4, dst=0),
            (1, 1): make_instr("ADD", src_a=0, src_b=0, dst=0),
        }
        mem = ConfigMemory(rows=2, cols=2, config_sequence=[c0, c1])
        # rf_depth=8 to allow register indices 0-7
        self.array = PEArray(rows=2, cols=2, rf_depth=8, config_memory=mem)

        pe00 = self.array.get_pe(0, 0)
        pe00.rf.write(0, 3)
        pe00.rf.write(1, 4)
        pe00.rf.write(2, 2)

        pe01 = self.array.get_pe(0, 1)
        pe01.rf.write(0, 5)
        pe01.rf.write(1, 6)
        pe01.rf.write(3, 10)

        pe10 = self.array.get_pe(1, 0)
        pe10.rf.write(0, 9)
        pe10.rf.write(4, 1)

        pe11 = self.array.get_pe(1, 1)
        pe11.rf.write(0, 1)
        pe11.rf.write(1, 1)

    def test_pe00_result(self) -> None:
        self.array.run(2)
        assert self.array.get_pe(0, 0).rf.read(0) == 14

    def test_pe01_result(self) -> None:
        self.array.run(2)
        assert self.array.get_pe(0, 1).rf.read(0) == 40

    def test_pe10_result(self) -> None:
        self.array.run(2)
        assert self.array.get_pe(1, 0).rf.read(0) == -8

    def test_pe11_result(self) -> None:
        self.array.run(2)
        assert self.array.get_pe(1, 1).rf.read(0) == 4

    def test_current_cycle_after_run(self) -> None:
        self.array.run(2)
        assert self.array.current_cycle == 2

    def test_activity_trace_length(self) -> None:
        self.array.run(2)
        assert len(self.array.activity_trace) == 2

    def test_all_pes_active_in_cycle0(self) -> None:
        self.array.run(2)
        cycle0 = self.array.activity_trace[0]
        for pe_id_str in ["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"]:
            assert cycle0[pe_id_str]["active"] is True

    def test_activity_summary_two_per_pe(self) -> None:
        self.array.run(2)
        for count in self.array.get_activity_summary().values():
            assert count == 2

    def test_reset_and_rerun_deterministic(self) -> None:
        self.array.run(2)
        first = self.array.get_pe(0, 0).rf.read(0)

        self.array.reset()
        pe00 = self.array.get_pe(0, 0)
        pe00.rf.write(0, 3)
        pe00.rf.write(1, 4)
        pe00.rf.write(2, 2)
        self.array.run(2)

        assert self.array.get_pe(0, 0).rf.read(0) == first == 14


# ---------------------------------------------------------------------------
# Scenario 2: Dataflow pipeline via NoC Mesh
# ---------------------------------------------------------------------------

class TestFullFlowNoCDataflow:
    """
    PE(0,0) produces 15 and PE(0,1) consumes it via the west NoC link.

    Cycle 0: PE(0,0) ADD(10,5)=15 written to rf[0]. output_value=15.
    Cycle 1: NoC propagates 15 to PE(0,1) west input (mux_sel=4).
             PE(0,1) ADD(west=15, rf[1]=3) = 18 written to rf[2].

    Reference model: 10 + 5 = 15; 15 + 3 = 18.
    """

    def setup_method(self) -> None:
        c0 = {
            (0, 0): make_instr("ADD", src_a=0, src_b=1, dst=0, mux_sel=0),
            (0, 1): NOP,
        }
        c1 = {
            (0, 0): NOP,
            (0, 1): make_instr("ADD", src_a=0, src_b=1, dst=2, mux_sel=4),
        }
        mem = ConfigMemory(rows=1, cols=2, config_sequence=[c0, c1])
        self.array = PEArray(rows=1, cols=2, config_memory=mem)

        pe00 = self.array.get_pe(0, 0)
        pe00.rf.write(0, 10)
        pe00.rf.write(1, 5)

        self.array.get_pe(0, 1).rf.write(1, 3)

    def test_pipeline_final_result(self) -> None:
        self.array.run(2)
        assert self.array.get_pe(0, 1).rf.read(2) == 18

    def test_pe00_output_value_after_cycle0(self) -> None:
        # After cycle 0 pe00 produced 15; output_value reflects last tick
        self.array.step()
        assert self.array.get_pe(0, 0).output_value == 15

    def test_noc_west_input_after_propagation(self) -> None:
        self.array.step()   # cycle 0: pe00 output_value=15
        self.array.step()   # cycle 1: noc routes, pe01 reads west=15
        assert self.array.get_pe(0, 1).neighbor_inputs["west"] == 15

    def test_topology_locked_during_run(self) -> None:
        self.array.step()
        assert self.array.noc.locked

    def test_topology_unlocked_after_reset(self) -> None:
        self.array.run(2)
        self.array.reset()
        assert not self.array.noc.locked


# ---------------------------------------------------------------------------
# Scenario 3: Torus topology dataflow
# ---------------------------------------------------------------------------

class TestFullFlowTorusDataflow:
    """
    In a 1x2 Torus, PE(0,1) east wraps to PE(0,0).

    Cycle 0: PE(0,1) MUL(3,4)=12 written to rf[0]. output_value=12.
    Cycle 1: NoC Torus propagates 12 to PE(0,0) east input (mux_sel=3).
             PE(0,0) ADD(east=12, rf[1]=8) = 20 written to rf[2].

    Reference model: 3*4=12; 12+8=20.
    """

    def setup_method(self) -> None:
        c0 = {
            (0, 0): NOP,
            (0, 1): make_instr("MUL", src_a=0, src_b=1, dst=0, mux_sel=0),
        }
        c1 = {
            (0, 0): make_instr("ADD", src_a=0, src_b=1, dst=2, mux_sel=3),
            (0, 1): NOP,
        }
        mem = ConfigMemory(rows=1, cols=2, config_sequence=[c0, c1])
        self.array = PEArray(
            rows=1, cols=2,
            topology=TOPOLOGY_TORUS,
            config_memory=mem,
        )

        pe01 = self.array.get_pe(0, 1)
        pe01.rf.write(0, 3)
        pe01.rf.write(1, 4)

        self.array.get_pe(0, 0).rf.write(1, 8)

    def test_torus_pipeline_final_result(self) -> None:
        self.array.run(2)
        assert self.array.get_pe(0, 0).rf.read(2) == 20

    def test_torus_topology_is_active(self) -> None:
        assert self.array.get_topology() == TOPOLOGY_TORUS

    def test_pe01_output_value_after_cycle0(self) -> None:
        # After cycle 0 pe01 produced 12; output_value reflects last tick
        self.array.step()
        assert self.array.get_pe(0, 1).output_value == 12

    def test_torus_east_input_propagated(self) -> None:
        self.array.step()   # cycle 0: pe01 output_value=12
        self.array.step()   # cycle 1: torus routes 12 to pe00 east
        assert self.array.get_pe(0, 0).neighbor_inputs["east"] == 12


# ---------------------------------------------------------------------------
# Scenario 4: Multi-cycle accumulation
# ---------------------------------------------------------------------------

class TestFullFlowMultiCycleAccumulation:
    """
    PE(0,0) counter: rf[0] += 1 every cycle. After N cycles rf[0]==N.

    Exercises run() over many cycles and SYRS-PER-001 compliance.
    """

    def _run_counter(self, rows: int, cols: int, n: int) -> int:
        seq = [
            {(0, 0): make_instr("ADD", src_a=0, src_b=1, dst=0)}
            for _ in range(n)
        ]
        mem = ConfigMemory(rows=rows, cols=cols, config_sequence=seq)
        array = PEArray(rows=rows, cols=cols, config_memory=mem)
        array.get_pe(0, 0).rf.write(0, 0)
        array.get_pe(0, 0).rf.write(1, 1)
        array.run(n)
        return array.get_pe(0, 0).rf.read(0)

    def test_counter_10_cycles(self) -> None:
        assert self._run_counter(1, 1, 10) == 10

    def test_counter_50_cycles(self) -> None:
        assert self._run_counter(1, 1, 50) == 50

    def test_counter_100_cycles_4x4(self) -> None:
        # SYRS-PER-001: 100 cycles on 4x4 must complete correctly
        assert self._run_counter(4, 4, 100) == 100

    def test_activity_count_matches_cycles(self) -> None:
        n = 20
        seq = [
            {(0, 0): make_instr("ADD", src_a=0, src_b=1, dst=0)}
            for _ in range(n)
        ]
        mem = ConfigMemory(rows=1, cols=1, config_sequence=seq)
        array = PEArray(rows=1, cols=1, config_memory=mem)
        array.get_pe(0, 0).rf.write(1, 1)
        array.run(n)
        assert array.get_activity_summary()[str((0, 0))] == n