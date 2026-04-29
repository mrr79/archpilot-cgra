"""
Modulo del Processing Element (PE) de la arquitectura CGRA.

El PE es la unidad fundamental de computo de la malla. Encapsula una
Unidad Funcional (FU), un Archivo de Registros (RF) y la logica de
multiplexion de entradas y salidas para la comunicacion con vecinos
a traves de la NoC.

Cumple: SYRS-FUN-002, SYRS-FUN-003, SYRS-FUN-004, SYRS-FUN-010,
        SYRS-MNT-001, SYRS-MNT-002, SYRS-QLY-002
"""

from typing import Optional

from archpilot_cgra.exceptions import (
    OverflowSimulationException,
    SimulationException,
)
from archpilot_cgra.functional_unit import FunctionalUnit
from archpilot_cgra.register_file import RegisterFile


# Direcciones validas para entradas de vecinos via NoC
VALID_DIRECTIONS: frozenset[str] = frozenset({"norte", "sur", "este", "oeste"})

# Mapa del selector del multiplexor de entrada (mux_sel)
# 0 = RF local, 1 = norte, 2 = sur, 3 = este, 4 = oeste
MUX_DIRECTION_MAP: dict[int, str] = {
    1: "norte",
    2: "sur",
    3: "este",
    4: "oeste",
}


class ProcessingElement:
    """
    Elemento de Procesamiento (PE) de la arquitectura CGRA.

    Integra una Unidad Funcional (FU), un Archivo de Registros (RF) y
    un multiplexor de entrada configurable, permitiendo que cada PE
    opere de forma autonoma en cada ciclo de reloj a partir de una
    instruccion distribuida por la Config Memory.

    En cada ciclo, el PE:
      1. Lee sus operandos desde el RF o desde entradas de vecinos (NoC).
      2. Ejecuta la operacion indicada por la instruccion actual.
      3. Escribe el resultado en el registro destino del RF.
      4. Detecta desbordamiento y lanza excepcion si es necesario.

    Cumple: SYRS-FUN-002, SYRS-FUN-010.

    Attributes:
        pe_id (tuple[int, int]):  identificador (fila, columna) en la malla.
        fu (FunctionalUnit):      unidad funcional del PE.
        rf (RegisterFile):        archivo de registros del PE.
        data_width (int):         ancho de bit de los datos.
        output_value (int):       ultimo resultado producido por el PE.
        input_mux_sel (int):      selector del mux de entrada A.
                                  0=RF, 1=norte, 2=sur, 3=este, 4=oeste.
        activity_count (int):     ciclos activos acumulados (traza SYRS-FUN-007).
        neighbor_inputs (dict):   valores de entrada de vecinos cardinales.

    Example:
        >>> pe = ProcessingElement(pe_id=(0, 0), rf_depth=4, data_width=32)
        >>> pe.rf.write(0, 10)
        >>> pe.rf.write(1, 5)
        >>> pe.load_instruction(
        ...     {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}
        ... )
        >>> pe.tick()
        15
        >>> pe.rf.read(2)
        15
    """

    def __init__(
        self,
        pe_id: tuple[int, int],
        rf_depth: int = 4,
        data_width: int = 32,
    ) -> None:
        """
        Inicializa el Processing Element.

        Args:
            pe_id (tuple[int, int]): identificador (fila, columna) en la malla.
            rf_depth (int):          profundidad del RF, entre 2 y 16.
                                     Por defecto 4.
            data_width (int):        ancho de bit para operandos y resultados.
                                     Por defecto 32 bits.

        Example:
            >>> pe = ProcessingElement(pe_id=(1, 2), rf_depth=8)
            >>> pe.pe_id
            (1, 2)
            >>> len(pe.rf)
            8
        """
        self.pe_id: tuple[int, int] = pe_id
        self.data_width: int = data_width
        self.fu: FunctionalUnit = FunctionalUnit(data_width=data_width)
        self.rf: RegisterFile = RegisterFile(depth=rf_depth, data_width=data_width)
        self.input_mux_sel: int = 0
        self.output_value: int = 0
        self.activity_count: int = 0
        self.neighbor_inputs: dict[str, int] = {
            "norte": 0,
            "sur": 0,
            "este": 0,
            "oeste": 0,
        }
        self._current_instruction: Optional[dict] = None

    # ------------------------------------------------------------------
    # Interfaz publica
    # ------------------------------------------------------------------

    def load_instruction(self, instruction: dict) -> None:
        """
        Carga la instruccion de configuracion para el proximo ciclo.

        La instruccion es un diccionario con los siguientes campos:

        - ``opcode``  (str): operacion a ejecutar.
        - ``src_a``   (int): indice RF del operando A (usado si mux_sel=0).
        - ``src_b``   (int): indice RF del operando B.
        - ``dst``     (int): indice RF donde se escribe el resultado.
        - ``mux_sel`` (int): selector del mux (0=RF, 1=norte, 2=sur,
          3=este, 4=oeste).

        Args:
            instruction (dict): diccionario de configuracion del ciclo.

        Example:
            >>> pe = ProcessingElement((0, 0))
            >>> pe.load_instruction(
            ...     {"opcode": "MUL", "src_a": 0, "src_b": 1,
            ...      "dst": 2, "mux_sel": 0}
            ... )
        """
        self._current_instruction = instruction
        self.input_mux_sel = instruction.get("mux_sel", 0)

    def tick(self) -> int:
        """
        Avanza la simulacion exactamente un ciclo de reloj.

        Secuencia de ejecucion:
          1. Leer operandos segun instruccion y selector del mux.
          2. Ejecutar la FU con los operandos obtenidos.
          3. Detectar desbordamiento (SYRS-FUN-010).
          4. Escribir resultado en el registro destino.
          5. Actualizar traza de actividad (SYRS-FUN-007).

        Returns:
            int: valor producido en este ciclo (almacenado en output_value).

        Raises:
            OverflowSimulationException: si el resultado supera el rango
                de data_width bits en complemento a dos.

        Example:
            >>> pe = ProcessingElement((0, 0), rf_depth=4, data_width=32)
            >>> pe.rf.write(0, 6)
            >>> pe.rf.write(1, 7)
            >>> pe.load_instruction(
            ...     {"opcode": "MUL", "src_a": 0, "src_b": 1, "dst": 3, "mux_sel": 0}
            ... )
            >>> pe.tick()
            42
        """
        if self._current_instruction is None:
            return 0

        instr = self._current_instruction
        opcode: str = instr.get("opcode", "NOP")
        src_a_idx: int = instr.get("src_a", 0)
        src_b_idx: int = instr.get("src_b", 0)
        dst_idx: int = instr.get("dst", 0)

        operand_a: int = self._read_mux(src_a_idx)
        operand_b: int = self.rf.read(src_b_idx)

        result: int = self.fu.execute(opcode, operand_a, operand_b)

        self._check_overflow(result)

        self.rf.write(dst_idx, result)
        self.output_value = result

        if self.fu.active:
            self.activity_count += 1

        return result

    def set_neighbor_input(self, direction: str, value: int) -> None:
        """
        Establece el valor de entrada proveniente de un vecino via NoC.

        Args:
            direction (str): cardinal del vecino: "norte", "sur", "este",
                             "oeste".
            value (int):     valor de datos recibido del vecino.

        Raises:
            SimulationException: si direction no es un cardinal valido.

        Example:
            >>> pe = ProcessingElement((1, 1))
            >>> pe.set_neighbor_input("norte", 99)
            >>> pe.neighbor_inputs["norte"]
            99
        """
        if direction not in VALID_DIRECTIONS:
            raise SimulationException(
                f"Direccion invalida: '{direction}'. "
                f"Validas: {sorted(VALID_DIRECTIONS)}."
            )
        self.neighbor_inputs[direction] = value

    def get_state(self) -> dict:
        """
        Retorna el estado completo del PE para inspeccion o depuracion.

        Util para el modo interactivo (SYRS-OPS-002) y para la
        construccion de la traza de actividad (SYRS-FUN-007).

        Returns:
            dict: con las claves: pe_id, registers, output_value,
                  last_opcode, activity_count, input_mux_sel,
                  neighbor_inputs.

        Example:
            >>> pe = ProcessingElement((0, 0))
            >>> state = pe.get_state()
            >>> list(state.keys())
            ['pe_id', 'registers', 'output_value', 'last_opcode',
             'activity_count', 'input_mux_sel', 'neighbor_inputs']
        """
        return {
            "pe_id": self.pe_id,
            "registers": self.rf.get_state(),
            "output_value": self.output_value,
            "last_opcode": self.fu.last_opcode,
            "activity_count": self.activity_count,
            "input_mux_sel": self.input_mux_sel,
            "neighbor_inputs": dict(self.neighbor_inputs),
        }

    def reset(self) -> None:
        """
        Reinicia el PE al estado inicial de simulacion.

        Limpia el RF, reinicia la FU, pone output_value y
        activity_count en 0 y descarta la instruccion actual.

        Example:
            >>> pe = ProcessingElement((0, 0))
            >>> pe.rf.write(0, 99)
            >>> pe.reset()
            >>> pe.rf.read(0)
            0
        """
        self.rf.reset()
        self.fu.reset()
        self.output_value = 0
        self.activity_count = 0
        self._current_instruction = None
        self.neighbor_inputs = {"norte": 0, "sur": 0, "este": 0, "oeste": 0}

    # ------------------------------------------------------------------
    # Metodos internos
    # ------------------------------------------------------------------

    def _read_mux(self, src_a_idx: int) -> int:
        """
        Lee el operando A segun el selector del multiplexor.

        Si mux_sel es 0, lee del RF local en src_a_idx.
        Si mux_sel es 1-4, lee de la entrada del vecino correspondiente.

        Args:
            src_a_idx (int): indice RF a usar cuando mux_sel == 0.

        Returns:
            int: valor del operando A seleccionado.
        """
        if self.input_mux_sel == 0:
            return self.rf.read(src_a_idx)
        direction = MUX_DIRECTION_MAP.get(self.input_mux_sel)
        if direction is None:
            return self.rf.read(src_a_idx)
        return self.neighbor_inputs[direction]

    def _check_overflow(self, result: int) -> None:
        """
        Verifica que el resultado no supere el rango de data_width bits.

        Para un entero con signo de N bits el rango es
        [-(2^(N-1)), 2^(N-1) - 1]. Si se supera, lanza
        OverflowSimulationException con el estado actual del PE.

        Args:
            result (int): valor a verificar.

        Raises:
            OverflowSimulationException: si abs(result) > max_val.
        """
        max_val: int = (2 ** (self.data_width - 1)) - 1
        if abs(result) > max_val:
            raise OverflowSimulationException(
                pe_id=self.pe_id,
                value=result,
                max_value=max_val,
                pe_state=self.get_state(),
            )
