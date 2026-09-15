from pathlib import Path
p = Path(__file__).resolve().parent.parent.parent / "modelo_predictivo_v5.py"; s = p.read_text(encoding="utf-8")
add = '''

# =============================================================================
# 7) Variante: posicion de lanza como palanca (lineal + cuadratica, signo libre)
# =============================================================================

LANZA = "posicion_vertical_lanza_mm"
VARIANTE_ACTIVA = "base"


def activar_variante_lanza(centro_mm: float = 2650.0) -> None:
    """Promueve `posicion_vertical_lanza_mm` (ACTION_primitiva del escalon) a palanca de CONTROL en los cinco modelos,
    con termino lineal y cuadratico centrado, ambos de SIGNO LIBRE: la convencion de la medida (mm hacia arriba o hacia
    abajo) no esta confirmada con planta, asi que la direccion y el eventual optimo interior se aprenden de los datos.
    La posicion previa sigue en el estado (punto de partida). Muta los diccionarios de configuracion del modulo."""
    global VARIANTE_ACTIVA, ACCIONES_PRIMITIVAS_V5, ACCIONES_OPTIMIZABLES_V5, FEATURES_SOPORTE_V5, PALANCAS_V5, SIGNO_TEORICO_V5
    CENTROS_CUADRATICOS[LANZA] = float(centro_mm)
    sq = f"{LANZA}__sq"
    PALANCAS_V5 = {k: [a for a in v if a not in (LANZA, sq)] + [LANZA, sq] for k, v in PALANCAS_V5.items()}
    SIGNO_TEORICO_V5 = {k: dict(v, **{LANZA: 0, sq: 0}) for k, v in SIGNO_TEORICO_V5.items()}
    ACCIONES_PRIMITIVAS_V5 = {f: [a for a in v if a != LANZA] + [LANZA] for f, v in ACCIONES_PRIMITIVAS_V5.items()}
    ACCIONES_OPTIMIZABLES_V5 = {f: [a for a in v if a != LANZA] + [LANZA] for f, v in ACCIONES_OPTIMIZABLES_V5.items()}
    FEATURES_SOPORTE_V5 = {f: [a for a in v if a != LANZA] + [LANZA] for f, v in FEATURES_SOPORTE_V5.items()}
    VARIANTE_ACTIVA = "lanza"
    _verificar_seguridad()
'''
s = s.rstrip() + add + "\n"
old = '''def fila_a_state_action_context(fase: str, row: pd.Series) -> tuple[dict, dict, dict]:
    state, action, context = mp3.fila_a_state_action_context(fase, row)'''
new = '''def fila_a_state_action_context(fase: str, row: pd.Series) -> tuple[dict, dict, dict]:
    state, action, context = mp3.fila_a_state_action_context(fase, row)
    for a in ACCIONES_PRIMITIVAS_V5[fase]:
        if a not in action:
            action[a] = row.get(a, np.nan)'''
assert old in s; s = s.replace(old, new)
# copias propias de los diccionarios de acciones (no mutar los de v4)
s = s.replace("ACCIONES_OPTIMIZABLES_V5 = mp4.ACCIONES_OPTIMIZABLES_V4\nACCIONES_PRIMITIVAS_V5 = mp4.ACCIONES_PRIMITIVAS_V4",
              "ACCIONES_OPTIMIZABLES_V5 = {f: list(v) for f, v in mp4.ACCIONES_OPTIMIZABLES_V4.items()}\nACCIONES_PRIMITIVAS_V5 = {f: list(v) for f, v in mp4.ACCIONES_PRIMITIVAS_V4.items()}")
assert "ACCIONES_OPTIMIZABLES_V5 = {f: list(v)" in s
p.write_text(s, encoding="utf-8"); print("ok")
