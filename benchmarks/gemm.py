"""
GEMM 4×4 benchmark for ArchPilot-CGRA (ACT-17).

Maps a 4×4 integer matrix multiplication C = A × B onto a 4×4 CGRA
mesh and validates the result against the Golden Model (SYRS-FUN-009,
SYRS-FUN-012).

Mapping strategy — data-parallel:
  Each PE(i, j) independently computes one output element C[i][j].
  All 16 PEs execute the same 7-cycle program in parallel, but with
  different data pre-loaded in their register files, so the full 4×4
  result is produced in a single 7-cycle pass.

Register file layout per PE(i, j)  [rf_depth = 16]:
  rf[0]  = A[i][0]    rf[4]  = B[0][j]
  rf[1]  = A[i][1]    rf[5]  = B[1][j]
  rf[2]  = A[i][2]    rf[6]  = B[2][j]
  rf[3]  = A[i][3]    rf[7]  = B[3][j]

Program (7 cycles, all PEs run the same opcodes):
  Cycle 0: MUL rf[0]  × rf[4]  → rf[8]   (A[i][0]·B[0][j])
  Cycle 1: MUL rf[1]  × rf[5]  → rf[9]   (A[i][1]·B[1][j])
  Cycle 2: MUL rf[2]  × rf[6]  → rf[10]  (A[i][2]·B[2][j])
  Cycle 3: MUL rf[3]  × rf[7]  → rf[11]  (A[i][3]·B[3][j])
  Cycle 4: ADD rf[8]  + rf[9]  → rf[12]  (partial sum)
  Cycle 5: ADD rf[10] + rf[11] → rf[13]  (partial sum)
  Cycle 6: ADD rf[12] + rf[13] → rf[14]  (C[i][j])

After 7 cycles: array.get_pe(i, j).rf.read(14) == C[i][j].

Complies with: SYRS-FUN-009, SYRS-FUN-012
"""

from __future__ import annotations

from dataclasses import dataclass

from archpilot_cgra import ConfigMemory, PEArray
from archpilot_cgra.golden_model import GoldenModel, ValidationResult

# Fixed dimensions for the 4×4 GEMM benchmark
MATRIX_SIZE: int = 4
NUM_CYCLES: int = 7        # 4 MUL + 3 ADD
RF_DEPTH: int = 16         # needs indices 0–14
RF_RESULT_IDX: int = 14    # rf[14] holds C[i][j] after the program


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GEMMResult:
    """
    Result of one GEMM benchmark run on the CGRA.

    Attributes:
        A:           Input matrix A (4×4).
        B:           Input matrix B (4×4).
        cgra_C:      Output matrix produced by the CGRA simulator.
        golden_C:    Reference output from GoldenModel.gemm().
        validation:  Element-wise comparison result.
        num_cycles:  Number of simulated cycles (always 7).

    Example:
        >>> result = run_gemm(A, B)
        >>> result.validation.passed
        True
    """

    A: list[list[int]]
    B: list[list[int]]
    cgra_C: list[list[int]]
    golden_C: list[list[int]]
    validation: ValidationResult
    num_cycles: int

    def summary(self) -> str:
        """Return a formatted summary of the benchmark run."""
        lines = [
            "╔══════════════════════════════════════════╗",
            "║      GEMM 4×4 Benchmark — ArchPilot      ║",
            "╚══════════════════════════════════════════╝",
            f"  Cycles simulated : {self.num_cycles}",
            f"  Matrix size      : {MATRIX_SIZE}×{MATRIX_SIZE}",
            "",
            "  CGRA result C = A × B:",
        ]
        for row in self.cgra_C:
            lines.append(f"    {row}")
        lines += [
            "",
            "  Golden Model reference:",
        ]
        for row in self.golden_C:
            lines.append(f"    {row}")
        lines += ["", self.validation.summary()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark implementation
# ---------------------------------------------------------------------------

def _build_program() -> list[dict[tuple[int, int], dict]]:
    """
    Build the 7-cycle ConfigMemory sequence for the GEMM benchmark.

    Every PE receives the same opcode each cycle; different results arise
    from the different data pre-loaded in each PE's register file.
    """
    def instr(opcode: str, src_a: int, src_b: int, dst: int) -> dict:
        return {"opcode": opcode, "src_a": src_a,
                "src_b": src_b, "dst": dst, "mux_sel": 0}

    all_pes = {
        (i, j): None
        for i in range(MATRIX_SIZE)
        for j in range(MATRIX_SIZE)
    }

    cycles = [
        # 4 parallel multiplications
        {pe: instr("MUL", 0,  4,  8) for pe in all_pes},   # A[i][0]·B[0][j]
        {pe: instr("MUL", 1,  5,  9) for pe in all_pes},   # A[i][1]·B[1][j]
        {pe: instr("MUL", 2,  6, 10) for pe in all_pes},   # A[i][2]·B[2][j]
        {pe: instr("MUL", 3,  7, 11) for pe in all_pes},   # A[i][3]·B[3][j]
        # 3 additions to accumulate the dot product
        {pe: instr("ADD", 8,  9,  12) for pe in all_pes},  # prod0 + prod1
        {pe: instr("ADD", 10, 11, 13) for pe in all_pes},  # prod2 + prod3
        {pe: instr("ADD", 12, 13, 14) for pe in all_pes},  # final sum → C[i][j]
    ]
    return cycles


def _load_data(
    array: PEArray,
    A: list[list[int]],
    B: list[list[int]],
) -> None:
    """
    Pre-load matrix data into each PE's register file.

    PE(i, j) receives:
      rf[0..3] = A[i][0..3]  (row i of A)
      rf[4..7] = B[0..3][j]  (column j of B)
    """
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            pe = array.get_pe(i, j)
            for k in range(MATRIX_SIZE):
                pe.rf.write(k, A[i][k])        # row of A
                pe.rf.write(4 + k, B[k][j])    # column of B


def _extract_result(array: PEArray) -> list[list[int]]:
    """
    Read the final result matrix from rf[14] of each PE.

    Returns C as a 4×4 list of lists.
    """
    return [
        [array.get_pe(i, j).rf.read(RF_RESULT_IDX)
         for j in range(MATRIX_SIZE)]
        for i in range(MATRIX_SIZE)
    ]


def run_gemm(
    A: list[list[int]],
    B: list[list[int]],
) -> GEMMResult:
    """
    Execute the GEMM 4×4 benchmark on the CGRA simulator.

    Sets up a 4×4 mesh, loads matrices A and B into the PE register
    files, runs the 7-cycle program, and validates the output against
    GoldenModel.gemm().

    Args:
        A: 4×4 integer matrix.
        B: 4×4 integer matrix.

    Returns:
        GEMMResult with CGRA output, Golden Model reference, and
        ValidationResult.

    Raises:
        ValueError: if A or B are not 4×4.

    Example:
        >>> A = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]  # identity
        >>> B = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        >>> result = run_gemm(A, B)
        >>> result.validation.passed
        True
    """
    if len(A) != MATRIX_SIZE or any(len(r) != MATRIX_SIZE for r in A):
        raise ValueError(f"A must be {MATRIX_SIZE}×{MATRIX_SIZE}.")
    if len(B) != MATRIX_SIZE or any(len(r) != MATRIX_SIZE for r in B):
        raise ValueError(f"B must be {MATRIX_SIZE}×{MATRIX_SIZE}.")

    # Build and load the program
    program = _build_program()
    mem = ConfigMemory(
        rows=MATRIX_SIZE,
        cols=MATRIX_SIZE,
        config_sequence=program,
    )
    array = PEArray(
        rows=MATRIX_SIZE,
        cols=MATRIX_SIZE,
        rf_depth=RF_DEPTH,
        config_memory=mem,
    )

    # Pre-load input data
    _load_data(array, A, B)

    # Simulate
    array.run(NUM_CYCLES)

    # Extract and validate
    cgra_C = _extract_result(array)
    golden_C = GoldenModel.gemm(A, B)
    validation = GoldenModel.validate_matrix(cgra_C, golden_C)

    return GEMMResult(
        A=A, B=B,
        cgra_C=cgra_C,
        golden_C=golden_C,
        validation=validation,
        num_cycles=NUM_CYCLES,
    )


# ---------------------------------------------------------------------------
# Default test matrices (SYRS-FUN-009)
# ---------------------------------------------------------------------------

#: Standard test matrix A for the GEMM benchmark
DEFAULT_A: list[list[int]] = [
    [ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12],
    [13, 14, 15, 16],
]

#: Standard test matrix B for the GEMM benchmark
DEFAULT_B: list[list[int]] = [
    [16, 15, 14, 13],
    [12, 11, 10,  9],
    [ 8,  7,  6,  5],
    [ 4,  3,  2,  1],
]


def run_default_gemm() -> GEMMResult:
    """
    Run the standard GEMM benchmark with DEFAULT_A and DEFAULT_B.

    Convenience function used by the validation suite (SYRS-FUN-009).

    Example:
        >>> result = run_default_gemm()
        >>> result.validation.passed
        True
    """
    return run_gemm(DEFAULT_A, DEFAULT_B)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running GEMM 4×4 benchmark...\n")
    result = run_default_gemm()
    print(result.summary())