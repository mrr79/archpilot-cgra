"""
ArchPilot-CGRA: Modeling and Simulation Environment for CGRAs.

Main package. Exposes the public simulator API.

Modules (Week 9 - ACT-08):
  - exceptions:          simulator exception hierarchy.
  - functional_unit:     arithmetic Functional Unit (FU).
  - register_file:       per-PE Register File (RF).
  - processing_element:  complete Processing Element (PE).
"""

from archpilot_cgra.exceptions import (
    InvalidOperationException,
    OverflowSimulationException,
    SimulationException,
)
from archpilot_cgra.functional_unit import FunctionalUnit
from archpilot_cgra.processing_element import ProcessingElement
from archpilot_cgra.register_file import RegisterFile

__all__ = [
    "SimulationException",
    "OverflowSimulationException",
    "InvalidOperationException",
    "FunctionalUnit",
    "RegisterFile",
    "ProcessingElement",
]
