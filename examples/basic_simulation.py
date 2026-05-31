"""
Basic simulation example for ArchPilot-CGRA.

Demonstrates the minimal workflow:
  1. Create a 2×2 PE array.
  2. Load a two-cycle program via ConfigMemory.
  3. Initialise register values.
  4. Run the simulation.
  5. Inspect results and activity summary.

Run:
    python -m examples.basic_simulation
    # or from the repo root:
    python examples/basic_simulation.py
"""

from archpilot_cgra import ConfigMemory, PEArray


def main() -> None:
    # ── Program definition ────────────────────────────────────────────
    # Cycle 0: PE(0,0) computes ADD(rf[0], rf[1]) → rf[2]
    #          PE(0,1) computes MUL(rf[0], rf[1]) → rf[2]
    # Cycle 1: PE(0,0) reads rf[2] produced in cycle 0 and multiplies
    #          by rf[3] → rf[0]
    config_sequence = [
        {
            (0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1,
                     "dst": 2, "mux_sel": 0},
            (0, 1): {"opcode": "MUL", "src_a": 0, "src_b": 1,
                     "dst": 2, "mux_sel": 0},
        },
        {
            (0, 0): {"opcode": "MUL", "src_a": 2, "src_b": 3,
                     "dst": 0, "mux_sel": 0},
            (0, 1): {"opcode": "NOP", "src_a": 0, "src_b": 0,
                     "dst": 0, "mux_sel": 0},
        },
    ]

    mem = ConfigMemory(rows=2, cols=2, config_sequence=config_sequence)
    array = PEArray(rows=2, cols=2, rf_depth=8, config_memory=mem)

    # ── Register initialisation ───────────────────────────────────────
    pe00 = array.get_pe(0, 0)
    pe00.rf.write(0, 10)   # operand A
    pe00.rf.write(1,  5)   # operand B  →  ADD = 15
    pe00.rf.write(3,  3)   # multiplier →  MUL(15, 3) = 45

    pe01 = array.get_pe(0, 1)
    pe01.rf.write(0,  4)   # operand A
    pe01.rf.write(1,  7)   # operand B  →  MUL = 28

    # ── Simulation ───────────────────────────────────────────────────
    print("Running 2-cycle simulation on a 2×2 mesh...\n")
    history = array.run(num_cycles=2)

    for result in history:
        cycle = result["cycle"]
        print(f"Cycle {cycle}:")
        for pe_id, output in sorted(result["results"].items()):
            print(f"  PE{pe_id} output = {output}")
        print()

    # ── Final register values ─────────────────────────────────────────
    print("Final register values:")
    print(f"  PE(0,0) rf[0] = {pe00.rf.read(0)}"
          f"  (expected 45: ADD(10,5)=15 then MUL(15,3)=45)")
    print(f"  PE(0,1) rf[2] = {pe01.rf.read(2)}"
          f"  (expected 28: MUL(4,7)=28)")

    # ── Activity summary ─────────────────────────────────────────────
    print("\nActivity summary (active cycles per PE):")
    for pe_id_str, count in sorted(array.get_activity_summary().items()):
        print(f"  PE{pe_id_str}: {count} active cycle(s)")

    # ── Array state snapshot ─────────────────────────────────────────
    state = array.get_array_state()
    print(f"\nTotal simulated cycles: {state['current_cycle']}")
    print(f"Topology: {state['topology']}")


if __name__ == "__main__":
    main()