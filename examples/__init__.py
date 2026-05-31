"""
ArchPilot-CGRA: Modeling and Simulation Environment for CGRAs.

Main package. Exposes the complete public simulator API.

Modules:
  - exceptions:          simulator exception hierarchy.
  - functional_unit:     arithmetic Functional Unit (FU).
  - register_file:       per-PE Register File (RF).
  - processing_element:  complete Processing Element (PE).
  - config_memory:       centralized configuration memory.
  - noc:                 Network-on-Chip routing engine.
  - pe_array:            full mesh simulation engine.
  - ppa_engine:          PPA estimation engine (area, power, activity).
  - heatmap:             PE utilization heatmap generator.
"""

from archpilot_cgra.config_memory import ConfigMemory
from archpilot_cgra.exceptions import (
    ConfigMemoryException,
    InvalidOperationException,
    OverflowSimulationException,
    SimulationException,
)
from archpilot_cgra.functional_unit import FunctionalUnit
from archpilot_cgra.heatmap import HeatmapGenerator
from archpilot_cgra.noc import NoC, TOPOLOGY_MESH, TOPOLOGY_TORUS
from archpilot_cgra.pe_array import PEArray
from archpilot_cgra.ppa_engine import (
    ActivityCollector,
    AreaModel,
    PPAEngine,
    PowerAnalyzer,
    TechnologyParams,
)
from archpilot_cgra.processing_element import ProcessingElement
from archpilot_cgra.register_file import RegisterFile

__all__ = [
    # Core simulation engine
    "PEArray",
    "ProcessingElement",
    "FunctionalUnit",
    "RegisterFile",
    "ConfigMemory",
    "NoC",
    # Topology constants
    "TOPOLOGY_MESH",
    "TOPOLOGY_TORUS",
    # PPA Engine
    "PPAEngine",
    "TechnologyParams",
    "AreaModel",
    "ActivityCollector",
    "PowerAnalyzer",
    # Heatmap
    "HeatmapGenerator",
    # Exceptions
    "SimulationException",
    "OverflowSimulationException",
    "InvalidOperationException",
    "ConfigMemoryException",
]