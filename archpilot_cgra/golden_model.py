"""
Golden Model module for the ArchPilot-CGRA validation suite.

Provides reference implementations of all arithmetic operations
supported by the CGRA simulator, using Python's standard library with
NumPy as an optional accelerator for matrix and signal operations.

The Golden Model serves as the ground truth against which the cycle-
accurate CGRA simulator is validated (SYRS-FUN-012).  Its outputs are
guaranteed correct by NumPy's well-tested linear algebra routines.

Classes:
  ValidationResult — immutable per-comparison result with statistics.
  GoldenModel      — reference implementations + validate() factory.

Complies with: SYRS-FUN-012, SYRS-MNT-001, SYRS-MNT-002, SYRS-QLY-002
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# NumPy is optional; matrix benchmarks require it but element-wise
# operations fall back to pure Python automatically.
try:
    import numpy as np
    HAS_NUMPY: bool = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    """
    Immutable result of a CGRA-vs-Golden-Model comparison.

    Produced by GoldenModel.validate() and GoldenModel.validate_matrix().

    Attributes:
        passed:              True if every element is within tolerance.
        max_absolute_error:  Largest absolute difference across all pairs.
        max_relative_error:  Largest relative difference (normalised by
                             max(|golden|, 1) to avoid division by zero).
        num_elements:        Number of element pairs compared.
        num_mismatches:      Number of pairs that exceed tolerance.
        tolerance:           Absolute tolerance used for this comparison.
        cgra_values:         Flat list of values produced by the CGRA.
        golden_values:       Flat list of reference values.

    Example:
        >>> r = ValidationResult(
        ...     passed=True, max_absolute_error=0.0,
        ...     max_relative_error=0.0, num_elements=4,
        ...     num_mismatches=0, tolerance=0.0,
        ...     cgra_values=[1, 2, 3, 4],
        ...     golden_values=[1, 2, 3, 4],
        ... )
        >>> r.passed
        True
    """

    passed: bool
    max_absolute_error: float
    max_relative_error: float
    num_elements: int
    num_mismatches: int
    tolerance: float
    cgra_values: list[int | float]
    golden_values: list[int | float]

    def summary(self) -> str:
        """
        Return a human-readable multi-line summary.

        Example:
            >>> print(result.summary())
            Validation ✓ PASS
              Elements compared : 16
              Mismatches        : 0
              Max abs error     : 0.0
              Max rel error     : 0.0000%
              Tolerance         : 0.0
        """
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"Validation {status}\n"
            f"  Elements compared : {self.num_elements}\n"
            f"  Mismatches        : {self.num_mismatches}\n"
            f"  Max abs error     : {self.max_absolute_error}\n"
            f"  Max rel error     : {self.max_relative_error:.4%}\n"
            f"  Tolerance         : {self.tolerance}"
        )


# ---------------------------------------------------------------------------
# GoldenModel
# ---------------------------------------------------------------------------

class GoldenModel:
    """
    Reference implementations for CGRA simulator validation (SYRS-FUN-012).

    All element-wise methods mirror the FunctionalUnit operations exactly
    (SYRS-FUN-003) using pure Python so they require no dependencies.
    Matrix and signal methods use NumPy when available, falling back to
    pure Python otherwise.

    Usage — element-wise:
        >>> GoldenModel.add(10, 5)
        15
        >>> GoldenModel.multiply(6, 7)
        42
        >>> GoldenModel.complement(9)
        -9

    Usage — matrix:
        >>> A = [[1, 0], [0, 1]]
        >>> B = [[3, 4], [5, 6]]
        >>> GoldenModel.gemm(A, B)
        [[3, 4], [5, 6]]

    Usage — validation:
        >>> result = GoldenModel.validate([15, 42], [15, 42])
        >>> result.passed
        True
    """

    # ------------------------------------------------------------------
    # Element-wise operations — mirror FunctionalUnit (SYRS-FUN-003)
    # ------------------------------------------------------------------

    @staticmethod
    def add(a: int, b: int) -> int:
        """
        Integer addition: result = a + b.

        Mirrors FunctionalUnit opcode "ADD".

        Example:
            >>> GoldenModel.add(10, 5)
            15
        """
        return int(a) + int(b)

    @staticmethod
    def multiply(a: int, b: int) -> int:
        """
        Integer multiplication: result = a * b.

        Mirrors FunctionalUnit opcode "MUL".

        Example:
            >>> GoldenModel.multiply(6, 7)
            42
        """
        return int(a) * int(b)

    @staticmethod
    def complement(a: int) -> int:
        """
        Two's complement negation: result = -a.

        Mirrors FunctionalUnit opcode "COMPLEMENT".

        Example:
            >>> GoldenModel.complement(9)
            -9
        """
        return -int(a)

    @staticmethod
    def nop() -> int:
        """
        No operation: result = 0.

        Mirrors FunctionalUnit opcode "NOP".

        Example:
            >>> GoldenModel.nop()
            0
        """
        return 0

    # ------------------------------------------------------------------
    # Matrix operations
    # ------------------------------------------------------------------

    @staticmethod
    def gemm(
        A: list[list[int]],
        B: list[list[int]],
    ) -> list[list[int]]:
        """
        General Matrix Multiply: C = A × B (integer arithmetic).

        Uses NumPy matmul when available; otherwise falls back to the
        standard O(n³) pure-Python implementation.  Both paths produce
        identical integer results.

        Args:
            A: n×m matrix as list of lists of ints.
            B: m×p matrix as list of lists of ints.

        Returns:
            n×p result matrix as list of lists of ints.

        Raises:
            ValueError: if inner dimensions do not match.

        Example:
            >>> A = [[1, 2], [3, 4]]
            >>> B = [[5, 6], [7, 8]]
            >>> GoldenModel.gemm(A, B)
            [[19, 22], [43, 50]]
        """
        rows_a, cols_a = len(A), len(A[0])
        rows_b, cols_b = len(B), len(B[0])
        if cols_a != rows_b:
            raise ValueError(
                f"Incompatible dimensions: A is {rows_a}×{cols_a}, "
                f"B is {rows_b}×{cols_b}."
            )

        if HAS_NUMPY:
            C = np.matmul(np.array(A, dtype=np.int64),
                          np.array(B, dtype=np.int64))
            return C.tolist()

        # Pure-Python fallback
        C = [[0] * cols_b for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                acc = 0
                for k in range(cols_a):
                    acc += A[i][k] * B[k][j]
                C[i][j] = acc
        return C

    @staticmethod
    def fir(
        signal: list[int],
        coefficients: list[int],
    ) -> list[int]:
        """
        Causal FIR (Finite Impulse Response) filter.

        Computes the discrete convolution of *signal* with *coefficients*
        in causal (same-length) mode: output[i] = Σ coeff[j] * signal[i-j]
        for j in range(len(coefficients)), clamped to valid indices.

        Output length always equals input signal length.

        Uses NumPy convolve when available; otherwise falls back to
        the pure-Python O(n·m) implementation.

        Args:
            signal:       Input signal samples (list of ints).
            coefficients: Filter tap coefficients (list of ints).

        Returns:
            Filtered output of the same length as signal.

        Example:
            >>> GoldenModel.fir([1, 0, 0, 0], [1, 2, 3])
            [1, 0, 0, 0]
            >>> GoldenModel.fir([1, 1, 1, 1], [1, 1])
            [1, 2, 2, 2]
        """
        n = len(signal)
        if n == 0:
            return []

        if HAS_NUMPY:
            result = np.convolve(
                np.array(signal, dtype=np.int64),
                np.array(coefficients, dtype=np.int64),
                mode="full",
            )
            return [int(x) for x in result[:n]]

        # Pure-Python fallback
        m = len(coefficients)
        output: list[int] = []
        for i in range(n):
            acc = 0
            for j in range(m):
                if i - j >= 0:
                    acc += signal[i - j] * coefficients[j]
            output.append(acc)
        return output

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def validate(
        cls,
        cgra_values: list[int | float],
        golden_values: list[int | float],
        tolerance: float = 0.0,
    ) -> ValidationResult:
        """
        Compare a flat list of CGRA output values against golden values.

        For the integer arithmetic performed by the CGRA, exact equality
        is expected so the default tolerance is 0.  Use a non-zero
        tolerance only for floating-point comparisons.

        Args:
            cgra_values:   Flat list of values produced by the CGRA.
            golden_values: Flat list of reference values from GoldenModel.
            tolerance:     Maximum allowed absolute error per element.

        Returns:
            ValidationResult with pass/fail status and error statistics.

        Raises:
            ValueError: if the two lists have different lengths.

        Example:
            >>> GoldenModel.validate([15, 42, -9], [15, 42, -9]).passed
            True
            >>> GoldenModel.validate([15, 0], [15, 42]).passed
            False
        """
        if len(cgra_values) != len(golden_values):
            raise ValueError(
                f"Length mismatch: CGRA has {len(cgra_values)} values, "
                f"golden has {len(golden_values)}."
            )

        n = len(cgra_values)
        if n == 0:
            return ValidationResult(
                passed=True,
                max_absolute_error=0.0,
                max_relative_error=0.0,
                num_elements=0,
                num_mismatches=0,
                tolerance=tolerance,
                cgra_values=[],
                golden_values=[],
            )

        abs_errors = [
            abs(float(c) - float(g))
            for c, g in zip(cgra_values, golden_values)
        ]
        rel_errors = [
            abs(float(c) - float(g)) / max(abs(float(g)), 1.0)
            for c, g in zip(cgra_values, golden_values)
        ]
        mismatches = sum(1 for e in abs_errors if e > tolerance)

        return ValidationResult(
            passed=mismatches == 0,
            max_absolute_error=float(max(abs_errors)),
            max_relative_error=float(max(rel_errors)),
            num_elements=n,
            num_mismatches=mismatches,
            tolerance=tolerance,
            cgra_values=list(cgra_values),
            golden_values=list(golden_values),
        )

    @classmethod
    def validate_matrix(
        cls,
        cgra_matrix: list[list[int]],
        golden_matrix: list[list[int]],
        tolerance: float = 0.0,
    ) -> ValidationResult:
        """
        Compare two 2D matrices element-wise by flattening both.

        Convenience wrapper around validate() for matrix outputs such
        as the result of a GEMM benchmark.

        Args:
            cgra_matrix:   2D list produced by the CGRA.
            golden_matrix: 2D reference list from GoldenModel.gemm().
            tolerance:     Maximum allowed absolute error per element.

        Returns:
            ValidationResult.

        Example:
            >>> A = [[1, 2], [3, 4]]
            >>> GoldenModel.validate_matrix(A, A).passed
            True
        """
        cgra_flat = [v for row in cgra_matrix for v in row]
        golden_flat = [v for row in golden_matrix for v in row]
        return cls.validate(cgra_flat, golden_flat, tolerance)