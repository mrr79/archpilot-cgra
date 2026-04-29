"""
Modulo del Archivo de Registros (RF) del Processing Element.

El Archivo de Registros provee almacenamiento local de datos para cada PE.
Su profundidad es configurable de forma independiente por celda dentro del
rango definido por el estandar de la arquitectura.

Cumple: SYRS-FUN-004, SYRS-MNT-001, SYRS-MNT-002, SYRS-QLY-002
"""

from archpilot_cgra.exceptions import SimulationException


# Limites de profundidad del RF segun SYRS-FUN-004
RF_MIN_DEPTH: int = 2
RF_MAX_DEPTH: int = 16


class RegisterFile:
    """
    Archivo de Registros (RF) de un Processing Element de la CGRA.

    Provee almacenamiento de enteros con profundidad configurable entre
    2 y 16 registros por celda (SYRS-FUN-004). Todos los registros se
    inicializan en 0. Soporta lectura y escritura indexada con validacion
    de rango en cada acceso.

    Attributes:
        depth (int):      numero de registros disponibles [2, 16].
        data_width (int): ancho de bit de cada registro en bits.

    Example:
        >>> rf = RegisterFile(depth=4, data_width=32)
        >>> rf.write(0, 42)
        >>> rf.read(0)
        42
        >>> len(rf)
        4
    """

    def __init__(self, depth: int = 4, data_width: int = 32) -> None:
        """
        Inicializa el Archivo de Registros.

        Args:
            depth (int):      numero de registros. Debe estar en [2, 16].
                              Por defecto 4.
            data_width (int): ancho de bit de cada registro. Por defecto 32.

        Raises:
            SimulationException: si depth esta fuera del rango [2, 16].

        Example:
            >>> rf = RegisterFile(depth=8)
            >>> len(rf)
            8
            >>> RegisterFile(depth=1)
            Raises SimulationException
        """
        if not (RF_MIN_DEPTH <= depth <= RF_MAX_DEPTH):
            raise SimulationException(
                f"Profundidad de RF invalida: {depth}. "
                f"Debe estar entre {RF_MIN_DEPTH} y {RF_MAX_DEPTH}."
            )
        self.depth: int = depth
        self.data_width: int = data_width
        self._registers: list[int] = [0] * depth

    def read(self, index: int) -> int:
        """
        Lee el valor almacenado en el registro indicado.

        Args:
            index (int): indice del registro (0-based).

        Returns:
            int: valor entero almacenado en el registro.

        Raises:
            SimulationException: si index esta fuera del rango [0, depth-1].

        Example:
            >>> rf = RegisterFile(depth=4)
            >>> rf.write(2, 99)
            >>> rf.read(2)
            99
        """
        self._validate_index(index)
        return self._registers[index]

    def write(self, index: int, value: int) -> None:
        """
        Escribe un valor entero en el registro indicado.

        Args:
            index (int): indice del registro (0-based).
            value (int): valor entero a almacenar.

        Raises:
            SimulationException: si index esta fuera del rango [0, depth-1].

        Example:
            >>> rf = RegisterFile(depth=4)
            >>> rf.write(3, 7)
            >>> rf.read(3)
            7
        """
        self._validate_index(index)
        self._registers[index] = value

    def reset(self) -> None:
        """
        Pone todos los registros en 0.

        Se utiliza al reiniciar la simulacion o al inicializar el PE.

        Example:
            >>> rf = RegisterFile(depth=4)
            >>> rf.write(0, 55)
            >>> rf.reset()
            >>> rf.read(0)
            0
        """
        self._registers = [0] * self.depth

    def get_state(self) -> list[int]:
        """
        Retorna una copia del estado actual de todos los registros.

        Retorna una copia para evitar modificaciones externas al estado
        interno del archivo de registros.

        Returns:
            list[int]: lista con los valores actuales de los depth registros.

        Example:
            >>> rf = RegisterFile(depth=3)
            >>> rf.write(1, 5)
            >>> rf.get_state()
            [0, 5, 0]
        """
        return list(self._registers)

    def _validate_index(self, index: int) -> None:
        """
        Valida que el indice de registro sea accesible.

        Args:
            index (int): indice a validar.

        Raises:
            SimulationException: si index < 0 o index >= depth.
        """
        if not (0 <= index < self.depth):
            raise SimulationException(
                f"Indice de registro fuera de rango: {index}. "
                f"Rango valido: [0, {self.depth - 1}]."
            )

    def __len__(self) -> int:
        """Retorna la profundidad del archivo de registros."""
        return self.depth
