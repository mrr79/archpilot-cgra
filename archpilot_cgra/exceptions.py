"""
Modulo de excepciones del simulador ArchPilot-CGRA.

Define la jerarquia de excepciones utilizadas para senalizar
condiciones de error durante la simulacion, incluyendo
desbordamientos de datos y operaciones invalidas.

Cumple: SYRS-FUN-010, SYRS-MNT-001, SYRS-MNT-002, SYRS-QLY-002
"""


class SimulationException(Exception):
    """Excepcion base para todos los errores del simulador ArchPilot-CGRA."""


class OverflowSimulationException(SimulationException):
    """
    Se lanza cuando el resultado de una operacion supera el rango
    representable con el ancho de bit configurado en el PE.

    Cumple SYRS-FUN-010: el sistema genera una excepcion de simulacion
    y registra el estado del PE afectado al momento del desbordamiento.

    Attributes:
        pe_id (tuple[int, int]): identificador (fila, columna) del PE.
        value (int):             valor que causo el desbordamiento.
        max_value (int):         valor maximo permitido para data_width bits.
        pe_state (dict):         instantanea del estado del PE al momento
                                 del error.

    Example:
        >>> raise OverflowSimulationException(
        ...     pe_id=(0, 0), value=200, max_value=127,
        ...     pe_state={"registers": [100, 100, 0, 0]}
        ... )
        OverflowSimulationException: Overflow en PE(0, 0): ...
    """

    def __init__(
        self,
        pe_id: tuple,
        value: int,
        max_value: int,
        pe_state: dict,
    ) -> None:
        self.pe_id = pe_id
        self.value: int = value
        self.max_value: int = max_value
        self.pe_state: dict = pe_state
        super().__init__(
            f"Overflow en PE{pe_id}: valor={value} supera "
            f"max={max_value}. Estado: {pe_state}"
        )


class InvalidOperationException(SimulationException):
    """
    Se lanza cuando la Unidad Funcional recibe un opcode no soportado.

    Attributes:
        opcode (str): el opcode invalido que genero el error.

    Example:
        >>> raise InvalidOperationException("MODULO")
        InvalidOperationException: Operacion no soportada: 'MODULO'
    """

    def __init__(self, opcode: str) -> None:
        self.opcode: str = opcode
        super().__init__(f"Operacion no soportada: '{opcode}'")
