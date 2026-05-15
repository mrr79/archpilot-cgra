"""
ArchPilot-CGRA: Modeling and Simulation Environment for CGRAs.

Main package. Exposes the complete public simulator API.

Modules (Weeks 9-12):
  - exceptions:          simulator exception hierarchy.
  - functional_unit:     arithmetic Functional Unit (FU).
  - register_file:       per-PE Register File (RF).
  - processing_element:  complete Processing Element (PE).
  - config_memory:       centralized configuration memory (ACT-10).
  - noc:                 Network-on-Chip routing engine (ACT-11).
  - pe_array:            full mesh simulation engine (ACT-09/ACT-12).
"""

from archpilot_cgra.config_memory import ConfigMemory
from archpilot_cgra.exceptions import (
    ConfigMemoryException,
    InvalidOperationException,
    OverflowSimulationException,
    SimulationException,
)
from archpilot_cgra.functional_unit import FunctionalUnit
from archpilot_cgra.noc import NoC, TOPOLOGY_MESH, TOPOLOGY_TORUS
from archpilot_cgra.pe_array import PEArray
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
    # Exceptions
    "SimulationException",
    "OverflowSimulationException",
    "InvalidOperationException",
    "ConfigMemoryException",
]
