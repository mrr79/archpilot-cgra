"""
Modulo de la Unidad Funcional (FU) del Processing Element.

La Unidad Funcional es el componente aritmetico-logico principal de cada
PE. Soporta un conjunto de operaciones enteras parametrizables y mantiene
estado de actividad para la generacion de trazas de conmutacion.

Cumple: SYRS-FUN-003, SYRS-MNT-001, SYRS-MNT-002, SYRS-QLY-002
"""

from archpilot_cgra.exceptions import InvalidOperationException


# Conjunto de opcodes validos soportados por la FU
SUPPORTED_OPCODES: frozenset[str] = frozenset({"ADD", "COMPLEMENT", "MUL", "NOP"})


class FunctionalUnit:
    """
    Unidad Funcional (FU) de un Processing Element de la CGRA.

    Ejecuta operaciones aritmeticas enteras por ciclo de reloj. Soporta
    suma (ADD), complemento a dos (COMPLEMENT), multiplicacion (MUL) y
    no-operacion (NOP), cumpliendo SYRS-FUN-003.

    Mantiene el ultimo opcode ejecutado y un indicador de actividad
    para permitir la generacion de trazas de conmutacion (SYRS-FUN-007).

    Attributes:
        data_width (int): ancho de bit de los datos procesados.
        last_opcode (str): ultimo opcode ejecutado en la FU.
        active (bool):    True si la FU ejecuto una operacion real
                          (distinta de NOP) en el ultimo ciclo.

    Example:
        >>> fu = FunctionalUnit(data_width=32)
        >>> fu.execute("ADD", 10, 5)
        15
        >>> fu.execute("COMPLEMENT", 7)
        -7
        >>> fu.execute("MUL", 3, 4)
        12
        >>> fu.execute("NOP", 0)
        0
    """

    def __init__(self, data_width: int = 32) -> None:
        """
        Inicializa la Unidad Funcional.

        Args:
            data_width (int): ancho de bit de los operandos y resultado.
                              Por defecto 32 bits.

        Example:
            >>> fu = FunctionalUnit(data_width=16)
            >>> fu.data_width
            16
        """
        self.data_width: int = data_width
        self.last_opcode: str = "NOP"
        self.active: bool = False

    def execute(
        self,
        opcode: str,
        operand_a: int,
        operand_b: int = 0,
    ) -> int:
        """
        Ejecuta una operacion aritmetica sobre los operandos dados.

        Operaciones soportadas (SYRS-FUN-003):
          - ``ADD``:        resultado = operand_a + operand_b
          - ``COMPLEMENT``: resultado = -operand_a (complemento a dos)
          - ``MUL``:        resultado = operand_a * operand_b
          - ``NOP``:        resultado = 0, sin actividad registrada

        Args:
            opcode (str):    codigo de la operacion a ejecutar.
            operand_a (int): primer operando entero.
            operand_b (int): segundo operando entero.
                             Ignorado en COMPLEMENT y NOP. Por defecto 0.

        Returns:
            int: resultado entero de la operacion.

        Raises:
            InvalidOperationException: si el opcode no pertenece al
                conjunto de operaciones soportadas.

        Example:
            >>> fu = FunctionalUnit()
            >>> fu.execute("ADD", 3, 4)
            7
            >>> fu.execute("MUL", 6, 7)
            42
            >>> fu.execute("COMPLEMENT", 5)
            -5
        """
        if opcode not in SUPPORTED_OPCODES:
            raise InvalidOperationException(opcode)

        self.last_opcode = opcode
        self.active = opcode != "NOP"

        if opcode == "ADD":
            return operand_a + operand_b
        if opcode == "COMPLEMENT":
            return -operand_a
        if opcode == "MUL":
            return operand_a * operand_b
        return 0  # NOP

    def reset(self) -> None:
        """
        Reinicia el estado interno de la FU al estado inicial.

        Pone last_opcode en "NOP" y active en False. Se invoca al
        inicio de cada nuevo contexto de simulacion.

        Example:
            >>> fu = FunctionalUnit()
            >>> fu.execute("ADD", 1, 1)
            2
            >>> fu.reset()
            >>> fu.last_opcode
            'NOP'
            >>> fu.active
            False
        """
        self.last_opcode = "NOP"
        self.active = False
