"""Test anti-fuga (auditoria D): ninguna columna LEAKAGE/TARGET/DIAG ni salida de batch pasa el filtro de v7; todo el estado,
las palancas y las features de soporte de v7 son seguras. Ejecutar: .venv/Scripts/python.exe experimentos/v7/test_antifuga_v7.py"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "experimentos" / "v7"))
import pandas as pd
sys.path.insert(0, str(RAIZ / "experimentos" / "v6"))
import feature_engineering as fe
import masa_v6
import modelo_predictivo_v7 as mp7
import modelo_predictivo_v6 as mp6
import v7_lib as L

df = L.cargar_df()
roles = dict(fe.CLASIFICACION_FEATURES); roles.update(masa_v6.CLASIFICACION_M6)
inseguras_por_rol = {c for c, r in roles.items() if ("LEAKAGE" in r) or r in ("TARGET", "DIAG")}
salidas = [c for c in df.columns if any(k in c for k in ("sn_en_metal", "sn_en_dross", "sn_en_polvo", "rendimiento", "recuperacion", "k_anclaje", "anclada"))]
fallos = [c for c in df.columns if (c in inseguras_por_rol or c in salidas) and mp7.es_segura_v7(c)]
assert not fallos, f"columnas inseguras que pasan el filtro: {fallos}"
usadas = set()
for k, v in list(mp6.ESTADO_V6.items()) + list(mp6.PALANCAS_V6.items()) + list(mp6.FEATURES_SOPORTE_V6.items()):
    usadas |= set(v)
no_seguras = [c for c in usadas if not mp7.es_segura_v7(c)]
assert not no_seguras, f"features usadas por v7 no seguras: {no_seguras}"
contemporaneas = [c for c in usadas if c in ("m6_G_pct", "m6_masa_kg", "m6_sn_inv_kg", "m6_feo_inv_kg", "m6_resto_frac") or (c.startswith("ley_") and "escoria" in c and not c.endswith("_prev"))]
assert not contemporaneas, f"features contemporaneas en el estado: {contemporaneas}"
print(f"OK: {len(df.columns)} columnas revisadas; {len(inseguras_por_rol)} roles inseguros y {len(salidas)} salidas de batch bloqueadas; {len(usadas)} features de v7 seguras")
