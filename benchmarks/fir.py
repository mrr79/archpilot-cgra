"""
FIR filter benchmark for ArchPilot-CGRA (ACT-18).

Maps a 4-tap causal FIR filter onto a 1×4 CGRA mesh and validates
the output against the Golden Model (SYRS-FUN-009, SYRS-FUN-012).

A FIR (Finite Impulse Response) filter computes:

    y[n] = h[0]·x[n] + h[1]·x[n-1] + h[2]·x[n-2] + h[3]·x[n-3]

which is a dot product of the coefficient vector h with a sliding
window of the input signal x.  This is structurally identical to
one row of GEMM, so the CGRA program is the same 7-cycle sequence.

Mapping strategy — data-parallel:
  A 1×4 mesh is used (1 row, 4 columns).  Each PE(0, j) independently
  computes one output sample y[j + NUM_TAPS - 1], covering 4 consecutive
  fully-causal outputs from y[3] to y[6] for a 7-sample input signal.

Register file layout per PE(0, j):
  rf[0]  = h[0]       rf[4] = x[j + 3]   (x[n]   — current sample)
  rf[1]  = h[1]       rf[5] = x[j + 2]   (x[n-1])
  rf[2]  = h[2]       rf[6] = x[j + 1]   (x[n-2])
  rf[3]  = h[3]       rf[7] = x[j]       (x[n-3])

Program (7 cycles — same opcode structure as GEMM):
  Cycle 0: MUL rf[0] × rf[4] → rf[8]    (h[0]·x[n])
  Cycle 1: MUL rf[1] × rf[5] → rf[9]    (h[1]·x[n-1])
  Cycle 2: MUL rf[2] × rf[6] → rf[10]   (h[2]·x[n-2])
  Cycle 3: MUL rf[3] × rf[7] → rf[11]   (h[3]·x[n-3])
  Cycle 4: ADD rf[8]  + rf[9]  → rf[12]  (partial sum)
  Cycle 5: ADD rf[10] + rf[11] → rf[13]  (partial sum)
  Cycle 6: ADD rf[12] + rf[13] → rf[14]  (y[n])

After 7 cycles: array.get_pe(0, j).rf.read(14) == y[j + NUM_TAPS - 1].

Design note: both FIR and GEMM reduce to dot products, so the CGRA
achieves algorithm reuse — the same 7-cycle program handles both
benchmarks through reconfiguration of the input data only.

Complies with: SYRS-FUN-009, SYRS-FUN-012
"""

from __future__ import annotations

from dataclasses import dataclass

from archpilot_cgra import ConfigMemory, PEArray
from archpilot_cgra.golden_model import GoldenModel, ValidationResult

# Fixed benchmark parameters
NUM_TAPS: int = 4                           # filter coefficients
NUM_OUTPUTS: int = 4                        # output samples per run
SIGNAL_LENGTH: int = NUM_OUTPUTS + NUM_TAPS - 1  # = 7 (min signal length)
NUM_CYCLES: int = 7                         # 4 MUL + 3 ADD
RF_DEPTH: int = 16                          # needs indices 0-14
RF_RESULT_IDX: int = 14                     # holds y[n] after program
MESH_ROWS: int = 1
MESH_COLS: int = NUM_OUTPUTS               # 1×4 mesh


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FIRResult:
    """
    Result of one FIR benchmark run on the CGRA.

    Attributes:
        signal:        Input signal used.
        coefficients:  Filter tap coefficients.
        cgra_output:   Output samples y[3..6] produced by the CGRA.
        golden_output: Reference output from GoldenModel.fir()[3:7].
        validation:    Element-wise comparison result.
        num_cycles:    Number of simulated cycles (always 7).

    Example:
        >>> result = run_fir(signal, coeffs)
        >>> result.validation.passed
        True
    """

    signal: list[int]
    coefficients: list[int]
    cgra_output: list[int]
    golden_output: list[int]
    validation: ValidationResult
    num_cycles: int

    def summary(self) -> str:
        """Return a formatted summary of the benchmark run."""
        lines = [
            "╔══════════════════════════════════════════╗",
            "║      FIR 4-tap Benchmark — ArchPilot     ║",
            "╚══════════════════════════════════════════╝",
            f"  Cycles simulated : {self.num_cycles}",
            f"  Filter taps      : {NUM_TAPS}",
            f"  Output samples   : {NUM_OUTPUTS}",
            "",
            f"  Signal x         : {self.signal}",
            f"  Coefficients h   : {self.coefficients}",
            "",
            f"  CGRA output  y[3..6] : {self.cgra_output}",
            f"  Golden Model y[3..6] : {self.golden_output}",
            "",
            self.validation.summary(),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark implementation
# ---------------------------------------------------------------------------

def _build_program() -> list[dict[tuple[int, int], dict]]:
    """
    Build the 7-cycle ConfigMemory sequence for the FIR benchmark.

    Identical opcode structure to the GEMM benchmark — both are dot
    products.  The difference is in the data loaded into each PE.
    """
    def instr(opcode: str, src_a: int, src_b: int, dst: int) -> dict:
        return {"opcode": opcode, "src_a": src_a,
                "src_b": src_b, "dst": dst, "mux_sel": 0}

    all_pes = {(0, j): None for j in range(MESH_COLS)}

    return [
        {pe: instr("MUL", 0,  4,  8) for pe in all_pes},   # h[0]·x[n]
        {pe: instr("MUL", 1,  5,  9) for pe in all_pes},   # h[1]·x[n-1]
        {pe: instr("MUL", 2,  6, 10) for pe in all_pes},   # h[2]·x[n-2]
        {pe: instr("MUL", 3,  7, 11) for pe in all_pes},   # h[3]·x[n-3]
        {pe: instr("ADD", 8,  9,  12) for pe in all_pes},  # partial sum 1
        {pe: instr("ADD", 10, 11, 13) for pe in all_pes},  # partial sum 2
        {pe: instr("ADD", 12, 13, 14) for pe in all_pes},  # y[n]
    ]


def _load_data(
    array: PEArray,
    signal: list[int],
    coefficients: list[int],
) -> None:
    """
    Pre-load filter and signal data into each PE's register file.

    PE(0, j) computes y[j + NUM_TAPS - 1], so its signal window is:
      x[n]   = signal[j + NUM_TAPS - 1]
      x[n-1] = signal[j + NUM_TAPS - 2]
      x[n-2] = signal[j + NUM_TAPS - 3]
      x[n-3] = signal[j]
    """
    for j in range(MESH_COLS):
        pe = array.get_pe(0, j)
        n = j + NUM_TAPS - 1          # output index in the full signal
        for k in range(NUM_TAPS):
            pe.rf.write(k, coefficients[k])   # rf[0..3] = h
            pe.rf.write(NUM_TAPS + k, signal[n - k])  # rf[4..7] = x window


def _extract_result(array: PEArray) -> list[int]:
    """Read y[3..6] from rf[14] of each PE in the row."""
    return [array.get_pe(0, j).rf.read(RF_RESULT_IDX)
            for j in range(MESH_COLS)]


def run_fir(
    signal: list[int],
    coefficients: list[int],
) -> FIRResult:
    """
    Execute the FIR 4-tap benchmark on the CGRA simulator.

    Sets up a 1×4 mesh, loads the signal windows and coefficients,
    runs the 7-cycle program, and validates against GoldenModel.fir().

    Args:
        signal:       Input signal with at least SIGNAL_LENGTH (7) samples.
        coefficients: Exactly NUM_TAPS (4) filter coefficients.

    Returns:
        FIRResult with CGRA output, Golden Model reference, and
        ValidationResult.

    Raises:
        ValueError: if signal is shorter than SIGNAL_LENGTH or
                    len(coefficients) != NUM_TAPS.

    Example:
        >>> result = run_fir([1,2,3,4,5,6,7], [1,1,1,1])
        >>> result.validation.passed
        True
        >>> result.cgra_output
        [10, 14, 18, 22]
    """
    if len(signal) < SIGNAL_LENGTH:
        raise ValueError(
            f"Signal must have at least {SIGNAL_LENGTH} samples, "
            f"got {len(signal)}."
        )
    if len(coefficients) != NUM_TAPS:
        raise ValueError(
            f"Expected {NUM_TAPS} coefficients, got {len(coefficients)}."
        )

    mem = ConfigMemory(
        rows=MESH_ROWS,
        cols=MESH_COLS,
        config_sequence=_build_program(),
    )
    array = PEArray(
        rows=MESH_ROWS,
        cols=MESH_COLS,
        rf_depth=RF_DEPTH,
        config_memory=mem,
    )

    _load_data(array, signal, coefficients)
    array.run(NUM_CYCLES)

    cgra_out = _extract_result(array)

    # Golden reference: fully-causal outputs y[NUM_TAPS-1 .. NUM_TAPS-1+NUM_OUTPUTS-1]
    full_golden = GoldenModel.fir(signal, coefficients)
    golden_out = full_golden[NUM_TAPS - 1: NUM_TAPS - 1 + NUM_OUTPUTS]

    validation = GoldenModel.validate(cgra_out, golden_out)

    return FIRResult(
        signal=list(signal),
        coefficients=list(coefficients),
        cgra_output=cgra_out,
        golden_output=golden_out,
        validation=validation,
        num_cycles=NUM_CYCLES,
    )


# ---------------------------------------------------------------------------
# Default test data (SYRS-FUN-009)
# ---------------------------------------------------------------------------

#: Standard test signal for the FIR benchmark
DEFAULT_SIGNAL: list[int] = [1, 2, 3, 4, 5, 6, 7]

#: 4-tap moving-average filter (all coefficients = 1)
DEFAULT_COEFFICIENTS: list[int] = [1, 1, 1, 1]


def run_default_fir() -> FIRResult:
    """
    Run the standard FIR benchmark with DEFAULT_SIGNAL and
    DEFAULT_COEFFICIENTS (4-tap moving average).

    Expected output: y[3..6] = [10, 14, 18, 22].

    Example:
        >>> result = run_default_fir()
        >>> result.cgra_output
        [10, 14, 18, 22]
    """
    return run_fir(DEFAULT_SIGNAL, DEFAULT_COEFFICIENTS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running FIR 4-tap benchmark...\n")
    result = run_default_fir()
    print(result.summary())