"""
Pruebas unitarias para ProcessingElement.

Verifica la integracion de FU y RF, las operaciones aritmeticas,
el multiplexor de entradas, la deteccion de overflow y el reset.

Cumple: SYRS-FUN-002, SYRS-FUN-010, SYRS-REL-001
"""

import pytest

from archpilot_cgra.exceptions import (
    OverflowSimulationException,
    SimulationException,
)
from archpilot_cgra.processing_element import ProcessingElement


def make_instr(
    opcode: str = "NOP",
    src_a: int = 0,
    src_b: int = 1,
    dst: int = 2,
    mux_sel: int = 0,
) -> dict:
    """Fabrica una instruccion de prueba con valores por defecto."""
    return {
        "opcode": opcode,
        "src_a": src_a,
        "src_b": src_b,
        "dst": dst,
        "mux_sel": mux_sel,
    }


class TestProcessingElement:
    """Suite de pruebas para el Processing Element."""

    def setup_method(self) -> None:
        # data_width=8 facilita provocar overflows en pruebas (max=127)
        self.pe = ProcessingElement(pe_id=(0, 0), rf_depth=4, data_width=8)

    # --- Creacion ---

    def test_pe_id_stored_correctly(self) -> None:
        assert self.pe.pe_id == (0, 0)

    def test_has_functional_unit(self) -> None:
        assert self.pe.fu is not None

    def test_has_register_file(self) -> None:
        assert self.pe.rf is not None

    def test_rf_depth_respected(self) -> None:
        pe = ProcessingElement(pe_id=(0, 0), rf_depth=8)
        assert len(pe.rf) == 8

    # --- Operaciones ADD, COMPLEMENT, MUL ---

    def test_add_two_registers(self) -> None:
        self.pe.rf.write(0, 10)
        self.pe.rf.write(1, 5)
        self.pe.load_instruction(make_instr("ADD"))
        assert self.pe.tick() == 15

    def test_add_result_written_to_dst(self) -> None:
        self.pe.rf.write(0, 10)
        self.pe.rf.write(1, 5)
        self.pe.load_instruction(make_instr("ADD", dst=2))
        self.pe.tick()
        assert self.pe.rf.read(2) == 15

    def test_mul_two_registers(self) -> None:
        self.pe.rf.write(0, 3)
        self.pe.rf.write(1, 4)
        self.pe.load_instruction(make_instr("MUL"))
        assert self.pe.tick() == 12

    def test_complement_register(self) -> None:
        self.pe.rf.write(0, 7)
        self.pe.load_instruction(make_instr("COMPLEMENT"))
        assert self.pe.tick() == -7

    def test_nop_returns_zero(self) -> None:
        self.pe.load_instruction(make_instr("NOP"))
        assert self.pe.tick() == 0

    # --- output_value ---

    def test_output_value_updated_after_tick(self) -> None:
        self.pe.rf.write(0, 5)
        self.pe.rf.write(1, 5)
        self.pe.load_instruction(make_instr("ADD"))
        self.pe.tick()
        assert self.pe.output_value == 10

    # --- Actividad ---

    def test_activity_count_increments_on_real_op(self) -> None:
        self.pe.rf.write(0, 1)
        self.pe.rf.write(1, 2)
        self.pe.load_instruction(make_instr("ADD"))
        self.pe.tick()
        assert self.pe.activity_count == 1

    def test_activity_count_unchanged_on_nop(self) -> None:
        self.pe.load_instruction(make_instr("NOP"))
        self.pe.tick()
        assert self.pe.activity_count == 0

    # --- Overflow ---

    def test_overflow_raises_exception(self) -> None:
        # 100 + 100 = 200 > 127 (max para 8 bits con signo)
        self.pe.rf.write(0, 100)
        self.pe.rf.write(1, 100)
        self.pe.load_instruction(make_instr("ADD"))
        with pytest.raises(OverflowSimulationException):
            self.pe.tick()

    def test_overflow_exception_contains_pe_id(self) -> None:
        self.pe.rf.write(0, 100)
        self.pe.rf.write(1, 100)
        self.pe.load_instruction(make_instr("ADD"))
        with pytest.raises(OverflowSimulationException) as exc_info:
            self.pe.tick()
        assert exc_info.value.pe_id == (0, 0)

    def test_overflow_exception_contains_state(self) -> None:
        self.pe.rf.write(0, 100)
        self.pe.rf.write(1, 100)
        self.pe.load_instruction(make_instr("ADD"))
        with pytest.raises(OverflowSimulationException) as exc_info:
            self.pe.tick()
        assert "pe_id" in exc_info.value.pe_state

    # --- Multiplexor de entrada ---

    def test_mux_sel_zero_reads_rf(self) -> None:
        self.pe.rf.write(0, 20)
        self.pe.rf.write(1, 5)
        self.pe.load_instruction(make_instr("ADD", mux_sel=0))
        assert self.pe.tick() == 25

    def test_mux_sel_norte_reads_neighbor(self) -> None:
        self.pe.set_neighbor_input("norte", 30)
        self.pe.rf.write(1, 5)
        self.pe.load_instruction(make_instr("ADD", mux_sel=1))
        assert self.pe.tick() == 35

    def test_mux_sel_sur_reads_neighbor(self) -> None:
        self.pe.set_neighbor_input("sur", 10)
        self.pe.rf.write(1, 3)
        self.pe.load_instruction(make_instr("ADD", mux_sel=2))
        assert self.pe.tick() == 13

    # --- set_neighbor_input ---

    def test_set_valid_neighbor(self) -> None:
        self.pe.set_neighbor_input("oeste", 55)
        assert self.pe.neighbor_inputs["oeste"] == 55

    def test_set_invalid_direction_raises(self) -> None:
        with pytest.raises(SimulationException):
            self.pe.set_neighbor_input("arriba", 10)

    # --- get_state ---

    def test_get_state_has_required_keys(self) -> None:
        state = self.pe.get_state()
        required = {
            "pe_id", "registers", "output_value",
            "last_opcode", "activity_count",
            "input_mux_sel", "neighbor_inputs",
        }
        assert required.issubset(state.keys())

    # --- reset ---

    def test_reset_clears_registers(self) -> None:
        self.pe.rf.write(0, 99)
        self.pe.reset()
        assert self.pe.rf.read(0) == 0

    def test_reset_clears_activity(self) -> None:
        self.pe.rf.write(0, 1)
        self.pe.rf.write(1, 1)
        self.pe.load_instruction(make_instr("ADD"))
        self.pe.tick()
        self.pe.reset()
        assert self.pe.activity_count == 0

    def test_reset_clears_output_value(self) -> None:
        self.pe.rf.write(0, 5)
        self.pe.rf.write(1, 5)
        self.pe.load_instruction(make_instr("ADD"))
        self.pe.tick()
        self.pe.reset()
        assert self.pe.output_value == 0

    def test_no_instruction_tick_returns_zero(self) -> None:
        assert self.pe.tick() == 0
