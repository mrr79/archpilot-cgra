"""
Tests for the GEMM 4×4 benchmark (ACT-17).

Verifies that the CGRA data-parallel GEMM mapping produces results
identical to the Golden Model reference across multiple matrix inputs.

Complies with: SYRS-FUN-009, SYRS-FUN-012, SYRS-REL-001
"""

import pytest

from archpilot_cgra.golden_model import GoldenModel
from benchmarks.gemm import (
    DEFAULT_A,
    DEFAULT_B,
    MATRIX_SIZE,
    NUM_CYCLES,
    RF_RESULT_IDX,
    GEMMResult,
    run_default_gemm,
    run_gemm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def identity() -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(4)] for i in range(4)]


def zeros() -> list[list[int]]:
    return [[0] * 4 for _ in range(4)]


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------

class TestGEMMCorrectness:
    """CGRA output must match the Golden Model for every input."""

    def test_identity_times_matrix(self) -> None:
        I = identity()
        A = DEFAULT_A
        result = run_gemm(I, A)
        assert result.validation.passed
        assert result.cgra_C == A

    def test_matrix_times_identity(self) -> None:
        I = identity()
        A = DEFAULT_A
        result = run_gemm(A, I)
        assert result.validation.passed
        assert result.cgra_C == A

    def test_zero_matrix_left(self) -> None:
        Z = zeros()
        result = run_gemm(Z, DEFAULT_A)
        assert result.validation.passed
        assert result.cgra_C == Z

    def test_zero_matrix_right(self) -> None:
        Z = zeros()
        result = run_gemm(DEFAULT_A, Z)
        assert result.validation.passed
        assert result.cgra_C == Z

    def test_default_matrices(self) -> None:
        result = run_default_gemm()
        assert result.validation.passed

    def test_default_spot_check_c00(self) -> None:
        # C[0][0] = 1·16 + 2·12 + 3·8 + 4·4 = 16+24+24+16 = 80
        result = run_default_gemm()
        assert result.cgra_C[0][0] == 80

    def test_default_spot_check_c33(self) -> None:
        # C[3][3] = 13·13 + 14·9 + 15·5 + 16·1 = 169+126+75+16 = 386
        result = run_default_gemm()
        assert result.cgra_C[3][3] == 386

    def test_small_values(self) -> None:
        A = [[1, 2, 0, 0],
             [0, 1, 3, 0],
             [0, 0, 1, 4],
             [0, 0, 0, 1]]
        B = [[2, 0, 0, 0],
             [0, 3, 0, 0],
             [0, 0, 4, 0],
             [0, 0, 0, 5]]
        result = run_gemm(A, B)
        assert result.validation.passed
        # C[0][0] = 1*2 + 2*0 + 0*0 + 0*0 = 2
        assert result.cgra_C[0][0] == 2
        # C[1][1] = 0*0 + 1*3 + 3*0 + 0*0 = 3
        assert result.cgra_C[1][1] == 3

    def test_all_ones(self) -> None:
        ones = [[1] * 4 for _ in range(4)]
        result = run_gemm(ones, ones)
        assert result.validation.passed
        # Each element of C = sum of 4 ones = 4
        for i in range(4):
            for j in range(4):
                assert result.cgra_C[i][j] == 4

    def test_negative_values(self) -> None:
        A = [[-1, 2, -3, 4],
             [5, -6, 7, -8],
             [-9, 10, -11, 12],
             [13, -14, 15, -16]]
        B = [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        result = run_gemm(A, B)
        assert result.validation.passed
        assert result.cgra_C == A

    def test_cgra_matches_golden_model_exactly(self) -> None:
        # Zero tolerance: no rounding, no approximation
        result = run_default_gemm()
        assert result.validation.max_absolute_error == 0.0
        assert result.validation.num_mismatches == 0


# ---------------------------------------------------------------------------
# Benchmark metadata
# ---------------------------------------------------------------------------

class TestGEMMMetadata:
    """Verify simulation parameters and result structure."""

    def test_num_cycles_is_seven(self) -> None:
        result = run_default_gemm()
        assert result.num_cycles == NUM_CYCLES
        assert result.num_cycles == 7

    def test_result_is_4x4(self) -> None:
        result = run_default_gemm()
        assert len(result.cgra_C) == MATRIX_SIZE
        assert all(len(row) == MATRIX_SIZE for row in result.cgra_C)

    def test_result_is_immutable(self) -> None:
        result = run_default_gemm()
        with pytest.raises((AttributeError, TypeError)):
            result.num_cycles = 99  # type: ignore[misc]

    def test_validation_result_type(self) -> None:
        from archpilot_cgra.golden_model import ValidationResult
        result = run_default_gemm()
        assert isinstance(result.validation, ValidationResult)

    def test_summary_contains_pass(self) -> None:
        result = run_default_gemm()
        assert "PASS" in result.summary()

    def test_summary_contains_cycles(self) -> None:
        result = run_default_gemm()
        assert str(NUM_CYCLES) in result.summary()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestGEMMInputValidation:
    """Invalid input should raise ValueError before touching the CGRA."""

    def test_non_square_A_raises(self) -> None:
        A_bad = [[1, 2, 3]] * 3   # 3×3, not 4×4
        with pytest.raises(ValueError):
            run_gemm(A_bad, DEFAULT_B)

    def test_non_square_B_raises(self) -> None:
        B_bad = [[1, 2]] * 4      # 4×2, not 4×4
        with pytest.raises(ValueError):
            run_gemm(DEFAULT_A, B_bad)

    def test_wrong_row_count_raises(self) -> None:
        A_bad = [[1, 2, 3, 4]] * 3   # only 3 rows
        with pytest.raises(ValueError):
            run_gemm(A_bad, DEFAULT_B)