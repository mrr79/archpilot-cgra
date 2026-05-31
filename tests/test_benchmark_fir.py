"""
Tests for the FIR 4-tap benchmark (ACT-18).

Verifies that the CGRA data-parallel FIR mapping produces results
identical to the Golden Model reference across multiple signal and
coefficient inputs.

Complies with: SYRS-FUN-009, SYRS-FUN-012, SYRS-REL-001
"""

import pytest

from archpilot_cgra.golden_model import GoldenModel, ValidationResult
from benchmarks.fir import (
    DEFAULT_COEFFICIENTS,
    DEFAULT_SIGNAL,
    NUM_CYCLES,
    NUM_OUTPUTS,
    NUM_TAPS,
    SIGNAL_LENGTH,
    FIRResult,
    run_default_fir,
    run_fir,
)


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------

class TestFIRCorrectness:
    """CGRA output must match the Golden Model for every valid input."""

    def test_default_benchmark_passes(self) -> None:
        result = run_default_fir()
        assert result.validation.passed

    def test_default_output_values(self) -> None:
        # Moving average: y[3]=1+2+3+4=10, y[4]=2+3+4+5=14,
        #                 y[5]=3+4+5+6=18, y[6]=4+5+6+7=22
        result = run_default_fir()
        assert result.cgra_output == [10, 14, 18, 22]

    def test_identity_coefficients(self) -> None:
        # h=[1,0,0,0]: output = current sample only
        signal = [3, 5, 7, 9, 11, 13, 15]
        coeffs = [1, 0, 0, 0]
        result = run_fir(signal, coeffs)
        assert result.validation.passed
        # y[3]=9, y[4]=11, y[5]=13, y[6]=15
        assert result.cgra_output == [9, 11, 13, 15]

    def test_delay_one_coefficient(self) -> None:
        # h=[0,1,0,0]: output = one-sample delayed input
        signal = [3, 5, 7, 9, 11, 13, 15]
        coeffs = [0, 1, 0, 0]
        result = run_fir(signal, coeffs)
        assert result.validation.passed
        # y[3]=7, y[4]=9, y[5]=11, y[6]=13
        assert result.cgra_output == [7, 9, 11, 13]

    def test_constant_signal(self) -> None:
        # All same value: y[n] = value * sum(h)
        signal = [5] * 7
        coeffs = [1, 2, 1, 0]   # sum = 4
        result = run_fir(signal, coeffs)
        assert result.validation.passed
        assert all(v == 5 * 4 for v in result.cgra_output)

    def test_zero_signal(self) -> None:
        signal = [0] * 7
        coeffs = [3, 2, 1, 4]
        result = run_fir(signal, coeffs)
        assert result.validation.passed
        assert result.cgra_output == [0, 0, 0, 0]

    def test_zero_coefficients(self) -> None:
        signal = [1, 2, 3, 4, 5, 6, 7]
        coeffs = [0, 0, 0, 0]
        result = run_fir(signal, coeffs)
        assert result.validation.passed
        assert result.cgra_output == [0, 0, 0, 0]

    def test_negative_signal_values(self) -> None:
        signal = [-1, -2, -3, -4, -5, -6, -7]
        coeffs = [1, 1, 1, 1]
        result = run_fir(signal, coeffs)
        assert result.validation.passed
        # Same as default but negated: [-10,-14,-18,-22]
        assert result.cgra_output == [-10, -14, -18, -22]

    def test_negative_coefficients(self) -> None:
        signal = [1, 2, 3, 4, 5, 6, 7]
        coeffs = [-1, -1, -1, -1]
        result = run_fir(signal, coeffs)
        assert result.validation.passed
        assert result.cgra_output == [-10, -14, -18, -22]

    def test_alternating_coefficients(self) -> None:
        # h=[1,-1,1,-1]: high-pass-like filter
        signal = [1, 2, 3, 4, 5, 6, 7]
        coeffs = [1, -1, 1, -1]
        result = run_fir(signal, coeffs)
        assert result.validation.passed
        # y[3] = 4 - 3 + 2 - 1 = 2
        # y[4] = 5 - 4 + 3 - 2 = 2
        assert result.cgra_output[0] == 2
        assert result.cgra_output[1] == 2

    def test_longer_signal_uses_first_window(self) -> None:
        # Only the first SIGNAL_LENGTH samples affect the output window
        signal = [1, 2, 3, 4, 5, 6, 7, 99, 99, 99]  # extra samples ignored
        coeffs = [1, 1, 1, 1]
        result = run_fir(signal, coeffs)
        expected = run_fir(signal[:7], coeffs)
        assert result.cgra_output == expected.cgra_output

    def test_cgra_matches_golden_exactly(self) -> None:
        result = run_default_fir()
        assert result.validation.max_absolute_error == 0.0
        assert result.validation.num_mismatches == 0

    def test_different_coefficients_different_output(self) -> None:
        signal = DEFAULT_SIGNAL
        r1 = run_fir(signal, [1, 1, 1, 1])
        r2 = run_fir(signal, [2, 1, 0, 1])
        assert r1.cgra_output != r2.cgra_output


# ---------------------------------------------------------------------------
# Benchmark metadata
# ---------------------------------------------------------------------------

class TestFIRMetadata:
    """Verify simulation parameters and result structure."""

    def test_num_cycles_is_seven(self) -> None:
        result = run_default_fir()
        assert result.num_cycles == NUM_CYCLES
        assert result.num_cycles == 7

    def test_output_length_is_num_outputs(self) -> None:
        result = run_default_fir()
        assert len(result.cgra_output) == NUM_OUTPUTS
        assert len(result.cgra_output) == 4

    def test_golden_output_length(self) -> None:
        result = run_default_fir()
        assert len(result.golden_output) == NUM_OUTPUTS

    def test_result_is_immutable(self) -> None:
        result = run_default_fir()
        with pytest.raises((AttributeError, TypeError)):
            result.num_cycles = 99  # type: ignore[misc]

    def test_validation_result_type(self) -> None:
        result = run_default_fir()
        assert isinstance(result.validation, ValidationResult)

    def test_summary_contains_pass(self) -> None:
        result = run_default_fir()
        assert "PASS" in result.summary()

    def test_summary_contains_output(self) -> None:
        result = run_default_fir()
        summary = result.summary()
        assert "[10, 14, 18, 22]" in summary

    def test_signal_stored_in_result(self) -> None:
        result = run_default_fir()
        assert result.signal == DEFAULT_SIGNAL

    def test_coefficients_stored_in_result(self) -> None:
        result = run_default_fir()
        assert result.coefficients == DEFAULT_COEFFICIENTS


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestFIRInputValidation:
    """Invalid input should raise ValueError before touching the CGRA."""

    def test_signal_too_short_raises(self) -> None:
        with pytest.raises(ValueError):
            run_fir([1, 2, 3], DEFAULT_COEFFICIENTS)  # < 7 samples

    def test_wrong_num_coefficients_raises(self) -> None:
        with pytest.raises(ValueError):
            run_fir(DEFAULT_SIGNAL, [1, 2, 3])  # need exactly 4

    def test_too_many_coefficients_raises(self) -> None:
        with pytest.raises(ValueError):
            run_fir(DEFAULT_SIGNAL, [1, 2, 3, 4, 5])  # need exactly 4

    def test_empty_signal_raises(self) -> None:
        with pytest.raises(ValueError):
            run_fir([], DEFAULT_COEFFICIENTS)

    def test_exactly_minimum_signal_length_ok(self) -> None:
        signal = [1] * SIGNAL_LENGTH   # exactly 7 samples
        result = run_fir(signal, [1, 0, 0, 0])
        assert result.validation.passed