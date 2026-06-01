"""
Unit tests for the PPA Engine module (ACT-19).

Verifies TechnologyParams defaults, AreaModel scaling, ActivityCollector
α-factor computation, PowerAnalyzer formulas, and the PPAEngine façade
including CSV export and SYRS-PER-002 timing constraint.

Complies with: SYRS-FUN-013, SYRS-USA-003, SYRS-PER-002, SYRS-REL-001
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from archpilot_cgra import ConfigMemory, PEArray
from archpilot_cgra.ppa_engine import (
    AreaModel,
    AreaReport,
    ActivityCollector,
    ActivityReport,
    PowerAnalyzer,
    PowerReport,
    PPAEngine,
    PPAReport,
    TechnologyParams,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_array(
    rows: int = 2,
    cols: int = 2,
    cycles: int = 10,
    active: bool = True,
) -> PEArray:
    """Create a PEArray with simulated activity for testing."""
    opcode = "ADD" if active else "NOP"
    instr = {"opcode": opcode, "src_a": 0, "src_b": 1,
             "dst": 0, "mux_sel": 0}
    seq = [{(r, c): instr for r in range(rows) for c in range(cols)}
           for _ in range(cycles)]
    mem = ConfigMemory(rows=rows, cols=cols, config_sequence=seq)
    array = PEArray(rows=rows, cols=cols, config_memory=mem)
    if active:
        for r in range(rows):
            for c in range(cols):
                array.get_pe(r, c).rf.write(1, 1)
    array.run(cycles)
    return array


# ---------------------------------------------------------------------------
# TechnologyParams
# ---------------------------------------------------------------------------

class TestTechnologyParams:
    """Tests for TechnologyParams dataclass."""

    def test_default_node_is_28nm(self) -> None:
        assert TechnologyParams().node_nm == 28

    def test_default_vdd(self) -> None:
        assert TechnologyParams().vdd_v == 0.9

    def test_default_freq(self) -> None:
        assert TechnologyParams().freq_mhz == 500.0

    def test_default_fu_area_positive(self) -> None:
        assert TechnologyParams().fu_area_um2 > 0

    def test_default_rf_area_per_bit_positive(self) -> None:
        assert TechnologyParams().rf_area_um2_per_bit > 0

    def test_custom_freq(self) -> None:
        tech = TechnologyParams(freq_mhz=1000.0)
        assert tech.freq_mhz == 1000.0

    def test_custom_vdd(self) -> None:
        tech = TechnologyParams(vdd_v=1.0)
        assert tech.vdd_v == 1.0

    def test_all_power_params_positive(self) -> None:
        tech = TechnologyParams()
        assert tech.p_leak_fu_uw > 0
        assert tech.p_leak_rf_bit_uw > 0
        assert tech.p_leak_router_port_uw > 0

    def test_all_capacitance_params_positive(self) -> None:
        tech = TechnologyParams()
        assert tech.cl_fu_ff > 0
        assert tech.cl_rf_ff > 0
        assert tech.cl_router_ff > 0


# ---------------------------------------------------------------------------
# AreaModel
# ---------------------------------------------------------------------------

class TestAreaModel:
    """Tests for AreaModel.estimate()."""

    def setup_method(self) -> None:
        self.tech = TechnologyParams()

    def test_returns_area_report(self) -> None:
        result = AreaModel.estimate(2, 2, 4, 32, "mesh", self.tech)
        assert isinstance(result, AreaReport)

    def test_all_components_positive(self) -> None:
        r = AreaModel.estimate(2, 2, 4, 32, "mesh", self.tech)
        assert r.fu_area_um2 > 0
        assert r.rf_area_um2 > 0
        assert r.noc_area_um2 > 0
        assert r.total_area_um2 > 0

    def test_total_equals_sum_of_parts(self) -> None:
        r = AreaModel.estimate(2, 2, 4, 32, "mesh", self.tech)
        expected = r.fu_area_um2 + r.rf_area_um2 + r.noc_area_um2
        assert abs(r.total_area_um2 - expected) < 0.01

    def test_mm2_conversion(self) -> None:
        r = AreaModel.estimate(1, 1, 4, 32, "mesh", self.tech)
        assert abs(r.total_area_mm2 - r.total_area_um2 / 1_000_000) < 5e-7

    def test_larger_mesh_has_more_area(self) -> None:
        r2 = AreaModel.estimate(2, 2, 4, 32, "mesh", self.tech)
        r4 = AreaModel.estimate(4, 4, 4, 32, "mesh", self.tech)
        assert r4.total_area_um2 > r2.total_area_um2

    def test_torus_noc_larger_than_mesh(self) -> None:
        mesh = AreaModel.estimate(4, 4, 4, 32, "mesh", self.tech)
        torus = AreaModel.estimate(4, 4, 4, 32, "torus", self.tech)
        assert torus.noc_area_um2 > mesh.noc_area_um2

    def test_deeper_rf_more_area(self) -> None:
        r4 = AreaModel.estimate(2, 2, 4, 32, "mesh", self.tech)
        r8 = AreaModel.estimate(2, 2, 8, 32, "mesh", self.tech)
        assert r8.rf_area_um2 > r4.rf_area_um2

    def test_wider_datapath_more_fu_area(self) -> None:
        r32 = AreaModel.estimate(2, 2, 4, 32, "mesh", self.tech)
        r64 = AreaModel.estimate(2, 2, 4, 64, "mesh", self.tech)
        assert r64.fu_area_um2 > r32.fu_area_um2

    def test_1x1_mesh_valid(self) -> None:
        r = AreaModel.estimate(1, 1, 4, 32, "mesh", self.tech)
        assert r.total_area_um2 > 0

    def test_rf_bits_recorded(self) -> None:
        # 2×2 mesh, 4 regs, 32 bits = 4*32 = 128 bits per PE, 4 PEs = 512 total
        r = AreaModel.estimate(2, 2, 4, 32, "mesh", self.tech)
        assert r.total_rf_bits == 4 * 2 * 2 * 32

    def test_router_ports_recorded(self) -> None:
        r = AreaModel.estimate(2, 2, 4, 32, "mesh", self.tech)
        assert r.total_router_ports > 0

    def test_result_is_immutable(self) -> None:
        r = AreaModel.estimate(1, 1, 4, 32, "mesh", self.tech)
        with pytest.raises((AttributeError, TypeError)):
            r.fu_area_um2 = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ActivityCollector
# ---------------------------------------------------------------------------

class TestActivityCollector:
    """Tests for ActivityCollector.collect()."""

    def test_empty_trace_returns_zeros(self) -> None:
        r = ActivityCollector.collect([], 32)
        assert r.num_cycles == 0
        assert r.mean_alpha_fu == 0.0
        assert r.mean_alpha_switch == 0.0

    def test_empty_trace_returns_activity_report(self) -> None:
        r = ActivityCollector.collect([], 32)
        assert isinstance(r, ActivityReport)

    def test_all_active_cycles_alpha_fu_one(self) -> None:
        array = _make_array(rows=1, cols=1, cycles=10, active=True)
        r = ActivityCollector.collect(array.activity_trace, 32)
        assert r.mean_alpha_fu == 1.0

    def test_all_nop_cycles_alpha_fu_zero(self) -> None:
        array = _make_array(rows=1, cols=1, cycles=5, active=False)
        r = ActivityCollector.collect(array.activity_trace, 32)
        assert r.mean_alpha_fu == 0.0

    def test_num_cycles_matches_trace_length(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=20, active=True)
        r = ActivityCollector.collect(array.activity_trace, 32)
        assert r.num_cycles == 20

    def test_alpha_fu_keys_match_pes(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=5, active=True)
        r = ActivityCollector.collect(array.activity_trace, 32)
        assert len(r.alpha_fu) == 4  # 2×2 mesh

    def test_alpha_switch_between_zero_and_one(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=50, active=True)
        r = ActivityCollector.collect(array.activity_trace, 32)
        assert 0.0 <= r.mean_alpha_switch <= 1.0

    def test_single_cycle_alpha_switch_equals_alpha_fu(self) -> None:
        array = _make_array(rows=1, cols=1, cycles=1, active=True)
        r = ActivityCollector.collect(array.activity_trace, 32)
        # Single cycle: alpha_switch falls back to alpha_fu
        assert r.alpha_switch[str((0, 0))] == r.alpha_fu[str((0, 0))]

    def test_alpha_fu_values_in_range(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=10, active=True)
        r = ActivityCollector.collect(array.activity_trace, 32)
        for v in r.alpha_fu.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# PowerAnalyzer
# ---------------------------------------------------------------------------

class TestPowerAnalyzer:
    """Tests for PowerAnalyzer.analyze()."""

    def setup_method(self) -> None:
        self.tech = TechnologyParams()
        self.area = AreaModel.estimate(2, 2, 4, 32, "mesh", self.tech)

    def _zero_activity(self) -> ActivityReport:
        return ActivityReport(
            alpha_fu={}, alpha_switch={},
            mean_alpha_fu=0.0, mean_alpha_switch=0.0,
            num_cycles=10,
        )

    def _full_activity(self) -> ActivityReport:
        return ActivityReport(
            alpha_fu={}, alpha_switch={},
            mean_alpha_fu=1.0, mean_alpha_switch=1.0,
            num_cycles=10,
        )

    def test_returns_power_report(self) -> None:
        act = self._zero_activity()
        result = PowerAnalyzer.analyze(self.area, act, self.tech, 2, 2)
        assert isinstance(result, PowerReport)

    def test_zero_activity_zero_dynamic_power(self) -> None:
        act = self._zero_activity()
        result = PowerAnalyzer.analyze(self.area, act, self.tech, 2, 2)
        assert result.dynamic_power_uw == 0.0

    def test_static_power_always_positive(self) -> None:
        act = self._zero_activity()
        result = PowerAnalyzer.analyze(self.area, act, self.tech, 2, 2)
        assert result.static_power_uw > 0

    def test_full_activity_dynamic_positive(self) -> None:
        act = self._full_activity()
        result = PowerAnalyzer.analyze(self.area, act, self.tech, 2, 2)
        assert result.dynamic_power_uw > 0

    def test_total_equals_dynamic_plus_static(self) -> None:
        act = self._full_activity()
        result = PowerAnalyzer.analyze(self.area, act, self.tech, 2, 2)
        expected = result.dynamic_power_uw + result.static_power_uw
        assert abs(result.total_power_uw - expected) < 0.001

    def test_mw_conversion(self) -> None:
        act = self._full_activity()
        result = PowerAnalyzer.analyze(self.area, act, self.tech, 2, 2)
        assert abs(result.total_power_mw - result.total_power_uw / 1000) < 1e-9

    def test_higher_freq_more_dynamic_power(self) -> None:
        act = self._full_activity()
        tech_slow = TechnologyParams(freq_mhz=250.0)
        tech_fast = TechnologyParams(freq_mhz=1000.0)
        area_s = AreaModel.estimate(2, 2, 4, 32, "mesh", tech_slow)
        area_f = AreaModel.estimate(2, 2, 4, 32, "mesh", tech_fast)
        r_slow = PowerAnalyzer.analyze(area_s, act, tech_slow, 2, 2)
        r_fast = PowerAnalyzer.analyze(area_f, act, tech_fast, 2, 2)
        assert r_fast.dynamic_power_uw > r_slow.dynamic_power_uw

    def test_larger_mesh_more_static_power(self) -> None:
        act = self._zero_activity()
        area_2x2 = AreaModel.estimate(2, 2, 4, 32, "mesh", self.tech)
        area_4x4 = AreaModel.estimate(4, 4, 4, 32, "mesh", self.tech)
        r_2x2 = PowerAnalyzer.analyze(area_2x2, act, self.tech, 2, 2)
        r_4x4 = PowerAnalyzer.analyze(area_4x4, act, self.tech, 4, 4)
        assert r_4x4.static_power_uw > r_2x2.static_power_uw

    def test_result_is_immutable(self) -> None:
        act = self._zero_activity()
        result = PowerAnalyzer.analyze(self.area, act, self.tech, 2, 2)
        with pytest.raises((AttributeError, TypeError)):
            result.total_power_uw = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PPAEngine (façade)
# ---------------------------------------------------------------------------

class TestPPAEngine:
    """Tests for the PPAEngine façade."""

    def test_estimate_returns_ppa_report(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=10)
        report = PPAEngine(array).estimate()
        assert isinstance(report, PPAReport)

    def test_area_report_in_result(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=10)
        report = PPAEngine(array).estimate()
        assert isinstance(report.area, AreaReport)

    def test_activity_report_in_result(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=10)
        report = PPAEngine(array).estimate()
        assert isinstance(report.activity, ActivityReport)

    def test_power_report_in_result(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=10)
        report = PPAEngine(array).estimate()
        assert isinstance(report.power, PowerReport)

    def test_elapsed_under_5s(self) -> None:
        # SYRS-PER-002: PPA results in < 5 seconds
        array = _make_array(rows=4, cols=4, cycles=100)
        report = PPAEngine(array).estimate()
        assert report.elapsed_s < 5.0

    def test_elapsed_is_positive(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=10)
        report = PPAEngine(array).estimate()
        assert report.elapsed_s >= 0.0

    def test_num_cycles_matches_simulation(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=25)
        report = PPAEngine(array).estimate()
        assert report.activity.num_cycles == 25

    def test_empty_trace_works(self) -> None:
        array = PEArray(rows=2, cols=2)
        report = PPAEngine(array).estimate()
        assert report.activity.num_cycles == 0
        assert report.area.total_area_um2 > 0

    def test_custom_tech_params(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=10)
        tech = TechnologyParams(freq_mhz=1000.0)
        report = PPAEngine(array, tech).estimate()
        assert report.area.total_area_um2 > 0

    def test_summary_contains_area(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=10)
        report = PPAEngine(array).estimate()
        assert "AREA" in report.summary()

    def test_summary_contains_power(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=10)
        report = PPAEngine(array).estimate()
        assert "POWER" in report.summary()

    def test_summary_contains_syrs_per002(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=10)
        report = PPAEngine(array).estimate()
        assert "SYRS-PER-002" in report.summary()

    def test_report_is_immutable(self) -> None:
        array = _make_array(rows=2, cols=2, cycles=5)
        report = PPAEngine(array).estimate()
        with pytest.raises((AttributeError, TypeError)):
            report.elapsed_s = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PPAEngine.to_csv()
# ---------------------------------------------------------------------------

class TestPPAEngineCSV:
    """Tests for CSV export (SYRS-USA-003)."""

    def setup_method(self) -> None:
        self.array = _make_array(rows=2, cols=2, cycles=10)
        self.engine = PPAEngine(self.array)

    def test_csv_file_created(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        self.engine.to_csv(path)
        assert Path(path).exists()
        Path(path).unlink()

    def test_csv_has_header(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        self.engine.to_csv(path)
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "section" in header
        assert "metric" in header
        assert "value" in header
        assert "unit" in header
        Path(path).unlink()

    def test_csv_has_area_section(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        self.engine.to_csv(path)
        with open(path, newline="") as f:
            content = f.read()
        assert "area" in content
        Path(path).unlink()

    def test_csv_has_power_section(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        self.engine.to_csv(path)
        with open(path, newline="") as f:
            content = f.read()
        assert "power" in content
        Path(path).unlink()

    def test_csv_has_disclaimer(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        self.engine.to_csv(path)
        with open(path, newline="") as f:
            content = f.read()
        assert "Pre-RTL" in content
        Path(path).unlink()

    def test_csv_parent_dir_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "report.csv"
            self.engine.to_csv(str(path))
            assert path.exists()