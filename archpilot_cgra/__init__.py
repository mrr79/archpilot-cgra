"""
ArchPilot-CGRA: Entorno de Modelado y Simulacion para CGRAs.

Paquete principal. Expone la API publica del simulador.

Modulos del paquete (Semana 9 - ACT-08):
  - exceptions:        jerarquia de excepciones del simulador.
  - functional_unit:   Unidad Funcional (FU) aritmetica.
  - register_file:     Archivo de Registros (RF) por PE.
  - processing_element: Processing Element completo (PE).
"""

from archpilot_cgra.exceptions import (
    SimulationException,
    OverflowSimulationException,
    InvalidOperationException,
)
from archpilot_cgra.functional_unit import FunctionalUnit
from archpilot_cgra.register_file import RegisterFile
from archpilot_cgra.processing_element import ProcessingElement

__all__ = [
    "SimulationException",
    "OverflowSimulationException",
    "InvalidOperationException",
    "FunctionalUnit",
    "RegisterFile",
    "ProcessingElement",
]
