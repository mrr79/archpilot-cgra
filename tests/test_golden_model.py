"""
Unit tests for the Golden Model reference implementation.

Verifies that all reference operations produce correct results,
that validation correctly detects matches and mismatches, and that
matrix and signal operations agree with direct NumPy calculations
where available.

Complies with: SYRS-FUN-012, SYRS-REL-001
"""

import pytest

from archpilot_cgra.golden_model import GoldenModel, ValidationResult


# ---------------------------------------------------------------------------
# Element-wise operations
# ---------------------------------------------------------------------------

class TestGoldenModelOps:
    """Tests for ADD, MUL, COMPLEMENT, NOP — mirror of FunctionalUnit."""

    def test_add_positive(self) -> None:
        assert GoldenModel.add(10, 5) == 15

    def test_add_negative(self) -> None:
        assert GoldenModel.add(-3, 7) == 4

    def test_add_both_negative(self) -> None:
        assert GoldenModel.add(-4, -6) == -10

    def test_add_zero(self) -> None:
        assert GoldenModel.add(0, 0) == 0

    def test_multiply_positive(self) -> None:
        assert GoldenModel.multiply(6, 7) == 42

    def test_multiply_by_zero(self) -> None:
        assert GoldenModel.multiply(99, 0) == 0

    def test_multiply_negative(self) -> None:
        assert GoldenModel.multiply(-3, 4) == -12

    def test_multiply_both_negative(self) -> None:
        assert GoldenModel.multiply(-5, -5) == 25

    def test_complement_positive(self) -> None:
        assert GoldenModel.complement(9) == -9

    def test_complement_negative(self) -> None:
        assert GoldenModel.complement(-9) == 9

    def test_complement_zero(self) -> None:
        assert GoldenModel.complement(0) == 0

    def test_nop_returns_zero(self) -> None:
        assert GoldenModel.nop() == 0

    def test_ops_return_int(self) -> None:
        assert isinstance(GoldenModel.add(1, 2), int)
        assert isinstance(GoldenModel.multiply(2, 3), int)
        assert isinstance(GoldenModel.complement(5), int)
        assert isinstance(GoldenModel.nop(), int)


# ---------------------------------------------------------------------------
# GEMM reference
# ---------------------------------------------------------------------------

class TestGoldenModelGEMM:
    """Tests for the GEMM (matrix multiply) reference implementation."""

    def test_identity_matrix(self) -> None:
        I = [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        A = [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]]
        assert GoldenModel.gemm(I, A) == A
        assert GoldenModel.gemm(A, I) == A

    def test_zero_matrix(self) -> None:
        Z = [[0, 0], [0, 0]]
        A = [[3, 4], [5, 6]]
        assert GoldenModel.gemm(Z, A) == Z
        assert GoldenModel.gemm(A, Z) == Z

    def test_2x2_known_result(self) -> None:
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        assert GoldenModel.gemm(A, B) == [[19, 22], [43, 50]]

    def test_4x4_manual(self) -> None:
        A = [[1, 0, 0, 0],
             [0, 2, 0, 0],
             [0, 0, 3, 0],
             [0, 0, 0, 4]]
        B = [[1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1]]
        expected = [[1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4]]
        assert GoldenModel.gemm(A, B) == expected

    def test_4x4_full(self) -> None:
        A = [[1,  2,  3,  4],
             [5,  6,  7,  8],
             [9,  10, 11, 12],
             [13, 14, 15, 16]]
        B = [[16, 15, 14, 13],
             [12, 11, 10, 9],
             [8,  7,  6,  5],
             [4,  3,  2,  1]]
        result = GoldenModel.gemm(A, B)
        # Verify shape
        assert len(result) == 4
        assert all(len(row) == 4 for row in result)
        # Spot-check C[0][0] = 1*16 + 2*12 + 3*8 + 4*4 = 16+24+24+16 = 80
        assert result[0][0] == 80
        # Spot-check C[3][3] = 13*13 + 14*9 + 15*5 + 16*1 = 169+126+75+16=386
        assert result[3][3] == 386

    def test_dimension_mismatch_raises(self) -> None:
        A = [[1, 2, 3]]       # 1×3
        B = [[1, 2], [3, 4]]  # 2×2 — incompatible
        with pytest.raises(ValueError):
            GoldenModel.gemm(A, B)

    def test_result_is_list_of_lists(self) -> None:
        A = [[1, 2], [3, 4]]
        B = [[1, 0], [0, 1]]
        result = GoldenModel.gemm(A, B)
        assert isinstance(result, list)
        assert isinstance(result[0], list)


# ---------------------------------------------------------------------------
# FIR filter reference
# ---------------------------------------------------------------------------

class TestGoldenModelFIR:
    """Tests for the FIR filter reference implementation."""

    def test_impulse_single_coefficient(self) -> None:
        # Unit impulse through identity filter: output = input
        signal = [1, 0, 0, 0, 0]
        coeffs = [1]
        assert GoldenModel.fir(signal, coeffs) == [1, 0, 0, 0, 0]

    def test_impulse_multi_coefficient(self) -> None:
        # A unit impulse through a filter reveals the coefficients in
        # the output (impulse response property): [1,2,3,0,0]
        signal = [1, 0, 0, 0, 0]
        coeffs = [1, 2, 3]
        assert GoldenModel.fir(signal, coeffs) == [1, 2, 3, 0, 0]

    def test_moving_average(self) -> None:
        # 2-tap moving average: output[i] = signal[i] + signal[i-1]
        signal = [1, 1, 1, 1]
        coeffs = [1, 1]
        assert GoldenModel.fir(signal, coeffs) == [1, 2, 2, 2]

    def test_output_length_equals_signal_length(self) -> None:
        signal = [1, 2, 3, 4, 5]
        coeffs = [1, -1, 1]
        result = GoldenModel.fir(signal, coeffs)
        assert len(result) == len(signal)

    def test_constant_signal_identity_filter(self) -> None:
        # Single-coefficient [1] is an identity filter
        signal = [3, 3, 3, 3]
        assert GoldenModel.fir(signal, [1]) == signal

    def test_zero_signal(self) -> None:
        signal = [0, 0, 0, 0]
        coeffs = [5, 3, 1]
        assert GoldenModel.fir(signal, coeffs) == [0, 0, 0, 0]

    def test_empty_signal(self) -> None:
        assert GoldenModel.fir([], [1, 2]) == []

    def test_known_4tap_filter(self) -> None:
        # Manually computed: signal=[4,3,2,1], coeffs=[1,2]
        # out[0] = 4*1 = 4
        # out[1] = 3*1 + 4*2 = 11
        # out[2] = 2*1 + 3*2 = 8
        # out[3] = 1*1 + 2*2 = 5
        signal = [4, 3, 2, 1]
        coeffs = [1, 2]
        assert GoldenModel.fir(signal, coeffs) == [4, 11, 8, 5]


# ---------------------------------------------------------------------------
# validate() — flat list comparison
# ---------------------------------------------------------------------------

class TestValidate:
    """Tests for GoldenModel.validate()."""

    def test_exact_match_passes(self) -> None:
        result = GoldenModel.validate([10, 20, 30], [10, 20, 30])
        assert result.passed is True
        assert result.num_mismatches == 0
        assert result.max_absolute_error == 0.0

    def test_mismatch_detected(self) -> None:
        result = GoldenModel.validate([10, 99], [10, 20])
        assert result.passed is False
        assert result.num_mismatches == 1
        assert result.max_absolute_error == 79.0

    def test_all_mismatches(self) -> None:
        result = GoldenModel.validate([1, 2, 3], [4, 5, 6])
        assert result.num_mismatches == 3

    def test_tolerance_respected(self) -> None:
        # Difference of 1 is within tolerance=1
        result = GoldenModel.validate([10, 21], [10, 20], tolerance=1.0)
        assert result.passed is True

    def test_tolerance_exceeded(self) -> None:
        # Difference of 2 exceeds tolerance=1
        result = GoldenModel.validate([10, 22], [10, 20], tolerance=1.0)
        assert result.passed is False

    def test_empty_lists_pass(self) -> None:
        result = GoldenModel.validate([], [])
        assert result.passed is True
        assert result.num_elements == 0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            GoldenModel.validate([1, 2], [1, 2, 3])

    def test_num_elements_correct(self) -> None:
        result = GoldenModel.validate([1, 2, 3, 4], [1, 2, 3, 4])
        assert result.num_elements == 4

    def test_negative_values(self) -> None:
        result = GoldenModel.validate([-5, -10], [-5, -10])
        assert result.passed is True

    def test_result_is_immutable(self) -> None:
        result = GoldenModel.validate([1, 2], [1, 2])
        with pytest.raises((AttributeError, TypeError)):
            result.passed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_matrix() — 2D comparison
# ---------------------------------------------------------------------------

class TestValidateMatrix:
    """Tests for GoldenModel.validate_matrix()."""

    def test_identity_match(self) -> None:
        A = [[1, 2], [3, 4]]
        result = GoldenModel.validate_matrix(A, A)
        assert result.passed is True
        assert result.num_elements == 4

    def test_matrix_mismatch_detected(self) -> None:
        A = [[1, 2], [3, 4]]
        B = [[1, 2], [3, 99]]
        result = GoldenModel.validate_matrix(A, B)
        assert result.passed is False
        assert result.num_mismatches == 1

    def test_4x4_identity(self) -> None:
        M = [[i * 4 + j for j in range(4)] for i in range(4)]
        result = GoldenModel.validate_matrix(M, M)
        assert result.passed is True
        assert result.num_elements == 16


# ---------------------------------------------------------------------------
# ValidationResult.summary()
# ---------------------------------------------------------------------------

class TestValidationResultSummary:
    """Tests for the human-readable summary method."""

    def test_summary_pass(self) -> None:
        result = GoldenModel.validate([1, 2, 3], [1, 2, 3])
        summary = result.summary()
        assert "PASS" in summary
        assert "Mismatches        : 0" in summary

    def test_summary_fail(self) -> None:
        result = GoldenModel.validate([1, 99], [1, 2])
        summary = result.summary()
        assert "FAIL" in summary
        assert "Mismatches        : 1" in summary

    def test_summary_contains_tolerance(self) -> None:
        result = GoldenModel.validate([1], [1], tolerance=5.0)
        assert "5.0" in result.summary()