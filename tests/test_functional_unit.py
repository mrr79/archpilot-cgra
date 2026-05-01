"""
Unit tests for FunctionalUnit.

Verifies the three mandatory operations (SYRS-FUN-003), invalid opcode
handling, activity tracking, and reset behavior.

Complies with: SYRS-REL-001
"""

import pytest

from archpilot_cgra.exceptions import InvalidOperationException
from archpilot_cgra.functional_unit import FunctionalUnit


class TestFunctionalUnit:
    """Test suite for the Functional Unit."""

    def setup_method(self) -> None:
        self.fu = FunctionalUnit(data_width=32)

    # --- ADD ---

    def test_add_positive_operands(self) -> None:
        assert self.fu.execute("ADD", 3, 4) == 7

    def test_add_negative_operand(self) -> None:
        assert self.fu.execute("ADD", -5, 3) == -2

    def test_add_zeros(self) -> None:
        assert self.fu.execute("ADD", 0, 0) == 0

    # --- COMPLEMENT ---

    def test_complement_positive(self) -> None:
        assert self.fu.execute("COMPLEMENT", 5) == -5

    def test_complement_negative(self) -> None:
        assert self.fu.execute("COMPLEMENT", -5) == 5

    def test_complement_zero(self) -> None:
        assert self.fu.execute("COMPLEMENT", 0) == 0

    def test_complement_ignores_operand_b(self) -> None:
        assert self.fu.execute("COMPLEMENT", 3, 999) == -3

    # --- MUL ---

    def test_mul_positive(self) -> None:
        assert self.fu.execute("MUL", 6, 7) == 42

    def test_mul_by_zero(self) -> None:
        assert self.fu.execute("MUL", 99, 0) == 0

    def test_mul_negative(self) -> None:
        assert self.fu.execute("MUL", -3, 4) == -12

    # --- NOP ---

    def test_nop_returns_zero(self) -> None:
        assert self.fu.execute("NOP", 100, 200) == 0

    def test_nop_does_not_set_active(self) -> None:
        self.fu.execute("NOP", 1, 2)
        assert not self.fu.active

    # --- Activity tracking ---

    def test_add_sets_active_true(self) -> None:
        self.fu.execute("ADD", 1, 1)
        assert self.fu.active

    def test_last_opcode_tracked(self) -> None:
        self.fu.execute("MUL", 2, 3)
        assert self.fu.last_opcode == "MUL"

    # --- NOP after real operation resets active ---

    def test_nop_resets_active_to_false(self) -> None:
        # Verifies the transition active=True -> active=False after NOP.
        # test_nop_does_not_set_active only checks NOP from the initial state.
        self.fu.execute("ADD", 1, 1)
        assert self.fu.active  # sanity check
        self.fu.execute("NOP", 0)
        assert not self.fu.active

    # --- Invalid opcode ---

    def test_invalid_opcode_raises_exception(self) -> None:
        with pytest.raises(InvalidOperationException):
            self.fu.execute("MODULO", 10, 3)

    def test_invalid_opcode_message(self) -> None:
        with pytest.raises(InvalidOperationException, match="MODULO"):
            self.fu.execute("MODULO", 1, 1)

    # --- Reset ---

    def test_reset_clears_last_opcode(self) -> None:
        self.fu.execute("ADD", 1, 1)
        self.fu.reset()
        assert self.fu.last_opcode == "NOP"

    def test_reset_clears_active(self) -> None:
        self.fu.execute("ADD", 1, 1)
        self.fu.reset()
        assert not self.fu.active
