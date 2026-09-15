"""ST-03 -- identificacion de las palancas de CONTROL sobre los 36 candidatos st_<id> (PLM v4)
y cadena de signos vs evidencia de batch (experimentos/19).

Para cada candidato (20 Reduccion + 16 Fusion) y cada variante de palancas (primitiva: carbon/GN/
O2/aire tasas; cinetica: estructura carbon x estado / exceso_o2 agregando O2+aire+GN) se ajusta un
`modelo_predictivo_v4.ModeloPLM` (y = g(S) + theta.(a - E[a|S]), theta por OLS sobre residuos
out-of-fold de GroupKFold(5) por batch, IC95% bootstrap cluster por batch, n_boot=200) sobre DEV
(`modelo_prescriptivo.split_dev_lockbox`), excluyendo `es_primer_escalon_batch`.

Salidas (esta carpeta):
  st_03_theta.csv          -- theta por candidato x variante x palanca (+ signo teorico, IC, R2 OOF)
  st_03_cadena_signos.csv  -- sign_chain = sign(theta) * sign(rho(target,rendimiento)_DEV) vs
                              sign(coef exp19 palanca->rendimiento, DEV) -> coherente/incoherente/indeterminado
  st_03_resumen_cadena.csv -- conteo de palancas coherentes/incoherentes/indeterminadas por candidato x variante
  st_03_log.txt            -- log completo (tabla de signo teorico razonada, progreso, resumen final)
  figs/st_03_forest_<fase>.png   -- forest plot theta_1sd_rel por palanca (4 paneles), variante primitiva
  figs/st_03_heatmap_<fase>.png  -- heatmap candidato x palanca de la cadena de signos, variante primitiva
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RAIZ = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(HERE))

import modelo_predictivo_v4 as mp4          # noqa: E402
import modelo_prescriptivo as mp            # noqa: E402
import st_targets as st                     # noqa: E402

OUT = HERE
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)
LOG_PATH = OUT / "st_03_log.txt"

N_SPLITS = 5
N_BOOT = 200
SEED = mp4.SEED
MIN_ROWS = 30
MIN_BATCHES = N_SPLITS

_log_f = open(LOG_PATH, "w", encoding="utf-8")


def log(msg: str = "") -> None:
    print(msg)
    _log_f.write(str(msg) + "\n")
    _log_f.flush()


# =============================================================================
# 1) Palancas por variante (mismo orden posicional = mismo "rol" fisico en las 2 variantes)
# =============================================================================
ROLES = ["carbon", "gn", "o2", "aire"]

PALANCAS_PRIMITIVA = {
    "Reducción": ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
    "Fusión":    ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
}
PALANCAS_CINETICA = {
    "Reducción": ["Cx_sn_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    "Fusión":    ["relacion_C_Sn_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
}
VARIANTES = {"primitiva": PALANCAS_PRIMITIVA, "cinetica": PALANCAS_CINETICA}

ESTADO = {
    "Reducción": mp4.ESTADO_V4[("Reducción", "sn_kg")],
    "Fusión": mp4.ESTADO_V4[("Fusión", "sn_kg")],
}

# =============================================================================
# 2) Mapeo palanca (rol) -> decision de batch de experimentos/19 (M1_Fusion / M2_Reduccion,
#    kpi=rendimiento_proxy_batch, muestra=dev)
# =============================================================================
EXP19_MODELO = {"Reducción": "M2_Reduccion", "Fusión": "M1_Fusion"}
EXP19_VAR = {
    ("Reducción", "carbon"): "carbon_total_R", ("Reducción", "gn"): "gn_total_R",
    ("Reducción", "o2"): "exceso_o2_R", ("Reducción", "aire"): "exceso_o2_R",
    ("Fusión", "carbon"): "C_por_Sn_F", ("Fusión", "gn"): "gn_por_t_carga_F",
    ("Fusión", "o2"): "exceso_o2_F", ("Fusión", "aire"): "exceso_o2_F",
}

# =============================================================================
# 3) Signo teorico esperado (declarado y razonado) por GRUPO mecanistico, orden (carbon, gn, o2, aire),
#    sobre los targets st_<id> ya ORIENTADOS "mayor = mejor" (direccion de st_targets.REGISTRO aplicada).
#
#    Razonamiento (Guia_Ausmelt_Sn.md 3.3 ventana selectiva Sn/Fe, 5.2-5.3 basicidad/matriz, 7.4 balance
#    de Fe y dross, 8/10.7 respuesta dinamica por escalon, 11.3 corte de reduccion, 22 arrastre/polvo;
#    modelo_predictivo_v4.py SIGNO_TEORICO_V4 y docstring ~l.36 "carbon: +Sn, ~0 FeO (batch); GN: +Sn
#    pero +FeO, no selectivo (batch)"):
#
#    - extraccion (Sn que deja la escoria, kapp, avance, -residual): el carbon es el reductor primario
#      (SnO2 + 2C -> Sn + 2CO) -> sube extraccion. El GN aporta CO/H2 de combustion incompleta, un
#      reductor menos selectivo pero que tambien reduce SnO2 -> sube extraccion (SIGNO_TEORICO_V4
#      Reduccion/sn_kg: Cx_sn_v4 +1, GN +1, exceso_o2 -1). El O2 y el aire de lanza son oxidantes:
#      compiten por potencial redox con el carbon/GN y bajan la extraccion.
#      -> carbon +1, gn +1, o2 -1, aire -1
#
#    - sobre-reduccion (FeO reducido a Fe0 y sus proxies: fraccion retenida, FeO/Sn disponible, Fe
#      metalizado/Sn cargado; targets ya orientados como "menos sobre-reduccion = mejor"): el mismo
#      par reductor (carbon, GN) que extrae Sn tambien reduce FeO en exceso (FeO + C -> Fe + CO;
#      SIGNO_TEORICO_V4 Reduccion/feo_kg: carbon +1, GN +1 sobre el FeO RAW) -> negativo sobre el
#      target orientado (mas reduccion = peor). El O2/aire reponen potencial oxidante y protegen la
#      matriz fayalitica (Guia 7.4, 11.3) -> positivo sobre el target orientado.
#      -> carbon -1, gn -1, o2 +1, aire +1
#
#    - metal-equivalente (Sn extraido - lambda_Fe * FeO reducido, lambda in {0.30,0.60,1.0}): combina
#      extraccion (+carbon,+gn,-o2,-aire) con la PENALIDAD de sobre-reduccion en signo opuesto para
#      carbon y gn. Cual de los dos efectos domina es precisamente la pregunta empirica que fija
#      lambda_Fe (experimentos 15/19, lecciones iteracion 4): la teoria de primer orden NO determina
#      el signo neto -> se declara ambiguo (0) en las 4 palancas.
#      -> 0, 0, 0, 0
#
#    - selectividad (Sn extraido / FeO reducido, log-selectividad, utilizacion de carbon, log(Sn/FeO)
#      final y su caida): el carbon ataca preferentemente SnO2 (menor energia de activacion / ventana
#      de potencial redox mas favorable que FeO, Guia 3.3) -> sube la razon Sn/FeO. El GN, al ser un
#      reductor difuso de combustion (no dirigido a una especie), degrada la selectividad -> baja la
#      razon. El O2/aire oxidan ambas especies de forma similar (no hay evidencia teorica de que
#      favorezcan una sobre otra) -> signo ambiguo (0).
#      -> carbon +1, gn -1, o2 0, aire 0
#
#    - conservacion en Fusion (targets orientados a NO extraer Sn de forma prematura / retenerlo
#      como SnO2: -Sn extraido, -fraccion extraida, +log(Sn/FeO) final, +Sn retenido/carbon): el
#      carbon y el GN (reductores) atacan tambien el SnO2 de Fusion antes de tiempo -> negativo
#      (empeora conservacion). El O2/aire (oxidantes) protegen el SnO2 en la escoria -> positivo.
#      -> carbon -1, gn -1, o2 +1, aire +1
#
#    - matriz de Fusion (FeO formado/retenido, FeO retenido por corriente de Fe, %FeO final, IRF):
#      el Fe de mineral/dross se oxida a FeO bajo condiciones oxidantes (O2/aire) -> positivo; bajo
#      condiciones reductoras (carbon, GN) el Fe tiende a metalizarse en vez de quedar como FeO en la
#      matriz fayalitica (Guia 7.4, 9.5) -> negativo.
#      -> carbon -1, gn -1, o2 +1, aire +1
#
#    - termico_nivel (T media de Fusion, orientada a -T: minimizar): GN y O2 liberan calor de
#      combustion -> suben T -> negativo sobre el target orientado. El aire diluye la combustion con
#      N2 (menor entalpia liberada por Nm3 que O2 puro a igual caudal) -> tiende a bajar T -> positivo.
#      El carbon tiene un efecto termico de segundo orden (reaccion de reduccion es endotermica pero
#      el flujo de masa de carbon es chico frente al gas) -> ambiguo (0).
#      -> carbon 0, gn -1, o2 -1, aire +1
#
#    - polvo (temperatura de gases pre-BHF, proxy de arrastre, orientada a -T_gas: minimizar): GN y O2
#      liberan calor que calienta los gases de salida -> negativo. El aire tiene DOS canales opuestos
#      (dilucion/enfriamiento vs. mas volumen y velocidad de gas -> mas arrastre, Guia 22) que no se
#      pueden ordenar a priori -> ambiguo (0). El carbon es indirecto/chico -> ambiguo (0).
#      -> carbon 0, gn -1, o2 -1, aire 0
#
#    - termico_ventana / basicidad_ventana (cercania a un optimo interior: ventana termica R18/F08,
#      basicidad F06): el objetivo NO es monotono en ninguna palanca (el signo depende de si el
#      escalon esta por encima o por debajo del optimo) -> sin signo teorico de primer orden (0).
#      -> 0, 0, 0, 0
#
#    - exploratorio (direccion=0 en st_targets.REGISTRO: dT de Reduccion/Fusion, masa de escoria por
#      carga): sin signo a priori por diseno.
#      -> 0, 0, 0, 0
# =============================================================================
FAMILIA_SIGNO: dict[str, tuple[int, int, int, int]] = {
    "extraccion":        (+1, +1, -1, -1),
    "sobre-reduccion":   (-1, -1, +1, +1),
    "metal-equivalente": (0, 0, 0, 0),
    "selectividad":      (+1, -1, 0, 0),
    "conservacion":      (-1, -1, +1, +1),
    "matriz":            (-1, -1, +1, +1),
    "termico_nivel":     (0, -1, -1, +1),
    "polvo":             (0, -1, -1, 0),
    "termico_ventana":   (0, 0, 0, 0),
    "explor":            (0, 0, 0, 0),
}

GRUPO_ID: dict[str, str] = {
    # Reduccion (20)
    "R01_sn_ext": "extraccion", "R06_frac_sn": "extraccion", "R07_kapp_sn": "extraccion",
    "R12_d_avance": "extraccion", "R13_sn_residual": "extraccion",
    "R02_feo_ret": "sobre-reduccion", "R08_feo_ret_frac": "sobre-reduccion",
    "R16_feo_por_sn_disp": "sobre-reduccion", "R20_fe_metalizado_por_sn": "sobre-reduccion",
    "R03_sn_net_l03": "metal-equivalente", "R04_sn_net_l06": "metal-equivalente", "R05_sn_net_l10": "metal-equivalente",
    "R09_selectividad": "selectividad", "R10_log_selectividad": "selectividad", "R11_util_carbon": "selectividad",
    "R14_log_sn_feo_next": "selectividad", "R15_d_log_sn_feo": "selectividad",
    "R17_dT": "explor", "R18_T_ventana": "termico_ventana", "R19_T_gas_bhf": "polvo",
    # Fusion (16)
    "F01_sn_ext": "conservacion", "F02_frac_sn_ext": "conservacion", "F11_log_sn_feo_next": "conservacion",
    "F12_d_log_sn_feo": "conservacion", "F14_sn_ret_por_c": "conservacion", "F16_sn_ret_frac": "conservacion",
    "F03_feo_formado": "matriz", "F04_feo_ret_por_fe": "matriz", "F05_ley_feo_next": "matriz", "F10_irf_next": "matriz",
    "F06_B2_ventana": "termico_ventana", "F08_T_ventana": "termico_ventana",
    "F07_T_media": "termico_nivel", "F09_T_gas_bhf": "polvo",
    "F13_masa_por_carga": "explor", "F15_dT": "explor",
    # Añadidos tras ST-05/06 (candidatos GANADORES compuestos, ver README.md): R21 es una retencion
    # logaritmica de FeO (mismo mecanismo que R02/R08/R16/R20) -> grupo sobre-reduccion. R22 = SDI
    # (GANADOR Reduccion) = ln(Sn depletado) + 10*ln(FeO retenido): combina un termino de EXTRACCION
    # (mayor=mejor: carbon+, gn+, o2-, aire-) con un termino de SOBRE-REDUCCION 10x mas pesado (mayor=
    # mejor: carbon-, gn-, o2+, aire+), igual que R03/R04/R05 (metal-equivalente): el signo NETO de
    # primer orden no esta determinado por la teoria (aunque el peso w=10 sugiere que domina el
    # termino de FeO, no hay una regla estequiometrica que fije el punto de cruce) -> se declara
    # ambiguo (0) y se agrupa junto con metal-equivalente pese a que st_targets.py lo etiqueta
    # familia="selectividad" (esa etiqueta describe el PROPOSITO -- depletar Sn sin metalizar Fe --
    # no la estructura aditiva de dos terminos con signos opuestos en carbon/GN que es lo relevante
    # para el signo teorico de las 4 palancas).
    "R21_ln_feo_ret": "sobre-reduccion",
    "R22_sdi": "metal-equivalente",
}
_faltan_registro = sorted(set(st.REGISTRO) - set(GRUPO_ID))
if _faltan_registro:
    raise AssertionError(f"GRUPO_ID no cubre st.REGISTRO (candidatos sin grupo mecanistico declarado): {_faltan_registro}")


def tabla_signo_log(ids: list[str]) -> str:
    filas = []
    for id_ in ids:
        r = st.REGISTRO[id_]
        grupo = GRUPO_ID[id_]
        signos = FAMILIA_SIGNO[grupo]
        filas.append((id_, r["fase"], r["familia"], grupo, *signos))
    d = pd.DataFrame(filas, columns=["id", "fase", "familia_registro", "grupo_mecanistico", "carbon", "gn", "o2", "aire"])
    return d.to_string(index=False)


# =============================================================================
# 4) Datos
# =============================================================================
def preparar_datos(df: pd.DataFrame, fase: str, S: list[str], palancas: list[str], col: str,
                    batches_dev: set) -> pd.DataFrame | None:
    cols = list(dict.fromkeys(S + palancas + [col, "Batch"]))
    m = (df["fase_proceso"] == fase) & (~df["es_primer_escalon_batch"]) & (df["Batch"].isin(batches_dev))
    d = df.loc[m, cols].dropna().copy()
    if len(d) < MIN_ROWS or d["Batch"].nunique() < MIN_BATCHES:
        return None
    lo, hi = d[col].quantile([0.005, 0.995])
    d[col] = d[col].clip(lo, hi)
    return d


def main() -> None:
    t_ini = time.time()
    log(f"ST-03 identificacion de control -- inicio {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"N_SPLITS={N_SPLITS} N_BOOT={N_BOOT} SEED={SEED}")
    log("")

    log("Cargando datos (cache_df_st.pkl) y agregando columnas cineticas v4 (Cx_sn_v4, Cx_av_v4)...")
    df = pd.read_pickle(HERE / "cache_df_st.pkl")
    df = mp4.agregar_columnas_v4(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    log(f"df {df.shape}, batches DEV={len(batches_dev)}, LOCKBOX={len(batches_lockbox)}")

    # st_targets.py es una libreria COMPARTIDA por los subagentes ST-01..07 que corren en paralelo;
    # su REGISTRO puede crecer mientras este script corre (p.ej. los GANADORES R21/R22 anadidos tras
    # ST-05/06). Este script solo procesa los candidatos cuya columna st_<id> ya esta materializada en
    # el cache_df_st.pkl que se nos indico usar como fuente de datos -- los que se hayan anadido a
    # REGISTRO despues de generar ese cache se excluyen explicitamente (no se regenera el cache: no es
    # un archivo que este habilitado a modificar ni a re-ejecutar).
    IDS = sorted(id_ for id_ in st.REGISTRO if f"st_{id_}" in df.columns)
    ids_excluidos = sorted(set(st.REGISTRO) - set(IDS))
    log(f"Candidatos en st.REGISTRO: {len(st.REGISTRO)}; con columna st_<id> en cache_df_st.pkl: {len(IDS)}")
    if ids_excluidos:
        log(f"EXCLUIDOS (anadidos a st_targets.py despues de generar cache_df_st.pkl, sin columna disponible): {ids_excluidos}")
    log("")
    log("Tabla de signo teorico declarada (grupo mecanistico -> carbon, gn, o2, aire):")
    log(tabla_signo_log(IDS))
    log("")

    log("Cargando st_batch.csv (agregados de batch) para rho(target, rendimiento) en DEV...")
    batch = pd.read_csv(OUT / "st_batch.csv")
    batch_dev = batch[~batch["es_lockbox"]].copy()
    log(f"st_batch DEV n={len(batch_dev)}")

    log("Cargando experimentos/19_regresiones_kpi.csv (coef exp19, kpi=rendimiento_proxy_batch, muestra=dev)...")
    exp19 = pd.read_csv(RAIZ / "experimentos" / "19_regresiones_kpi.csv")
    exp19_dev = exp19[(exp19["muestra"] == "dev") & (exp19["kpi"] == "rendimiento_proxy_batch")]
    exp19_lookup: dict[tuple[str, str], dict] = {}
    for _, row in exp19_dev.iterrows():
        exp19_lookup[(row["modelo"], row["variable"])] = dict(
            coef=row["coef_por_sd"], ci_lo=row["ci_lo"], ci_hi=row["ci_hi"], p=row["p"], sig=bool(row["significativa"]))
    for fase in ["Reducción", "Fusión"]:
        for role in ROLES:
            key = (EXP19_MODELO[fase], EXP19_VAR[(fase, role)])
            if key not in exp19_lookup:
                raise AssertionError(f"falta {key} en experimentos/19_regresiones_kpi.csv")
    log("Coeficientes exp19 usados (modelo, variable) -> coef_por_sd [ci_lo, ci_hi] p:")
    for fase in ["Reducción", "Fusión"]:
        for role in ROLES:
            key = (EXP19_MODELO[fase], EXP19_VAR[(fase, role)])
            v = exp19_lookup[key]
            log(f"  {fase:10s} {role:6s} {key[1]:20s} coef={v['coef']:+.4f} [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['p']:.3f} sig={v['sig']}")
    log("")

    # rho(target -> rendimiento) en DEV, Spearman sobre st_batch
    rho_por_id: dict[str, dict] = {}
    for id_ in IDS:
        col = "st_" + id_
        if col not in batch_dev.columns:
            log(f"  [SKIP rho] {id_}: {col} no esta en st_batch.csv (mismo motivo que arriba); se omite del todo")
            continue
        x, y = batch_dev[col], batch_dev["rendimiento_proxy_batch"]
        rho, p = spearmanr(x, y, nan_policy="omit")
        rho_por_id[id_] = dict(rho=float(rho) if pd.notna(rho) else np.nan, p=float(p) if pd.notna(p) else np.nan,
                                n=int((x.notna() & y.notna()).sum()))
    log("rho de Spearman (st_<id> batch, rendimiento_proxy_batch) en DEV -- primeras filas:")
    log(pd.DataFrame(rho_por_id).T.round(4).to_string())
    log("")

    theta_rows = []
    n_fit, n_fail = 0, 0
    total = len(IDS) * len(VARIANTES)
    for fase in ["Reducción", "Fusión"]:
        S = ESTADO[fase]
        ids_fase = [id_ for id_ in IDS if st.REGISTRO[id_]["fase"] == fase]
        for id_ in ids_fase:
            r = st.REGISTRO[id_]
            col = "st_" + id_
            grupo = GRUPO_ID[id_]
            signos_rol = FAMILIA_SIGNO[grupo]
            for variante, palancas_por_fase in VARIANTES.items():
                palancas = palancas_por_fase[fase]
                signo_dict = {p: signos_rol[j] for j, p in enumerate(palancas)}
                t0 = time.time()
                try:
                    d = preparar_datos(df, fase, S, palancas, col, batches_dev)
                    if d is None:
                        log(f"[SKIP] {id_:26s} {variante:10s} -- datos insuficientes tras dropna/umbral batches")
                        continue
                    modelo = mp4.ModeloPLM(estado=S, palancas=palancas, target=col)
                    modelo.fit(d, n_splits=N_SPLITS, n_boot=N_BOOT, seed=SEED)
                    tabla = modelo.tabla_theta(signo_teorico=signo_dict)
                    sd_target = float(d[col].std())
                    for j, palanca in enumerate(palancas):
                        role = ROLES[j]
                        fila_t = tabla.loc[palanca]
                        theta_rows.append(dict(
                            id=id_, fase=fase, familia=r["familia"], grupo_mecanistico=grupo, variante=variante,
                            palanca=palanca, role=role,
                            theta_unidad=float(fila_t["theta_unidad"]), theta_1sd=float(fila_t["theta_1sd"]),
                            ci_lo_1sd=float(fila_t["ci_lo_1sd"]), ci_hi_1sd=float(fila_t["ci_hi_1sd"]),
                            significativo=bool(fila_t["significativo"]),
                            signo_teorico=int(fila_t["signo_teorico"]),
                            concuerda_teoria=fila_t["concuerda"] if pd.notna(fila_t["concuerda"]) else np.nan,
                            theta_1sd_rel=float(fila_t["theta_1sd"]) / sd_target if sd_target > 0 else np.nan,
                            sd_target=sd_target,
                            r2_oof_interno=float(modelo.r2_oof_interno),
                            n_obs=len(d), n_batches=int(d["Batch"].nunique()),
                        ))
                    n_fit += 1
                    log(f"[OK]   {id_:26s} {variante:10s} n={len(d):5d} batches={d['Batch'].nunique():3d} "
                        f"r2_oof={modelo.r2_oof_interno:+.3f} t={time.time()-t0:5.1f}s "
                        f"({n_fit+n_fail}/{total})")
                except Exception as e:
                    n_fail += 1
                    log(f"[FAIL] {id_:26s} {variante:10s} -- {type(e).__name__}: {e}")
                    log(traceback.format_exc(limit=3))
                # checkpoint cada 10 ajustes
                if (n_fit + n_fail) % 10 == 0:
                    pd.DataFrame(theta_rows).to_csv(OUT / "st_03_theta.csv", index=False)

    theta_df = pd.DataFrame(theta_rows)
    theta_df.to_csv(OUT / "st_03_theta.csv", index=False)
    log(f"\nst_03_theta.csv escrito: {theta_df.shape}, fits OK={n_fit} FAIL={n_fail}")
    log(f"Tiempo total ajustes: {(time.time()-t_ini)/60:.1f} min")

    # =========================================================================
    # 5) Cadena de signos
    # =========================================================================
    log("\nConstruyendo cadena de signos (sign(theta) x sign(rho) vs sign(coef exp19))...")
    cadena_rows = []
    for _, row in theta_df.iterrows():
        fase, role, id_, variante = row["fase"], row["role"], row["id"], row["variante"]
        rho_info = rho_por_id[id_]
        sign_theta = float(np.sign(row["theta_1sd"]))
        sign_rho = float(np.sign(rho_info["rho"])) if pd.notna(rho_info["rho"]) else 0.0
        sign_chain = sign_theta * sign_rho
        key19 = (EXP19_MODELO[fase], EXP19_VAR[(fase, role)])
        v19 = exp19_lookup[key19]
        sign_exp19 = float(np.sign(v19["coef"]))
        if row["significativo"] and sign_chain != 0 and sign_exp19 != 0:
            if sign_chain == sign_exp19:
                estado_cadena = "coherente"
            else:
                estado_cadena = "incoherente"
        else:
            estado_cadena = "indeterminado"
        cadena_rows.append(dict(
            id=id_, fase=fase, variante=variante, palanca=row["palanca"], role=role,
            theta_1sd=row["theta_1sd"], theta_significativo=row["significativo"], sign_theta=sign_theta,
            rho_target_rendimiento=rho_info["rho"], rho_p=rho_info["p"], sign_rho=sign_rho,
            sign_chain=sign_chain,
            exp19_variable=key19[1], exp19_coef_por_sd=v19["coef"], exp19_p=v19["p"], exp19_sig=v19["sig"],
            sign_exp19=sign_exp19,
            estado_cadena=estado_cadena,
            valor_heatmap={"coherente": 1, "incoherente": -1, "indeterminado": 0}[estado_cadena],
        ))
    cadena_df = pd.DataFrame(cadena_rows)
    cadena_df.to_csv(OUT / "st_03_cadena_signos.csv", index=False)
    log(f"st_03_cadena_signos.csv escrito: {cadena_df.shape}")

    resumen_rows = []
    for (id_, variante), g in cadena_df.groupby(["id", "variante"]):
        fase = g["fase"].iloc[0]
        cnt = g["estado_cadena"].value_counts()
        resumen_rows.append(dict(
            id=id_, fase=fase, variante=variante,
            n_coherente=int(cnt.get("coherente", 0)), n_incoherente=int(cnt.get("incoherente", 0)),
            n_indeterminado=int(cnt.get("indeterminado", 0)),
            coherente_carbon=g.loc[g["role"] == "carbon", "estado_cadena"].iloc[0] if (g["role"] == "carbon").any() else None,
            coherente_gn=g.loc[g["role"] == "gn", "estado_cadena"].iloc[0] if (g["role"] == "gn").any() else None,
        ))
    resumen_df = pd.DataFrame(resumen_rows).sort_values(["fase", "variante", "n_coherente"], ascending=[True, True, False])
    resumen_df.to_csv(OUT / "st_03_resumen_cadena.csv", index=False)
    log(f"st_03_resumen_cadena.csv escrito: {resumen_df.shape}")

    # =========================================================================
    # 6) Figuras
    # =========================================================================
    log("\nGenerando figuras...")
    for fase in ["Reducción", "Fusión"]:
        sub = theta_df[(theta_df["fase"] == fase) & (theta_df["variante"] == "primitiva")]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(20, max(4, 0.35 * sub["id"].nunique())), sharey=False)
        for ax, role in zip(axes, ROLES):
            s = sub[sub["role"] == role].sort_values("theta_1sd_rel")
            if s.empty:
                continue
            palanca_nombre = s["palanca"].iloc[0]
            colors = ["#2a9d8f" if sig else "#adb5bd" for sig in s["significativo"]]
            lo_rel = s["ci_lo_1sd"] / s["sd_target"]
            hi_rel = s["ci_hi_1sd"] / s["sd_target"]
            xerr = np.abs(np.vstack([s["theta_1sd_rel"] - lo_rel, hi_rel - s["theta_1sd_rel"]]))
            ax.errorbar(s["theta_1sd_rel"], range(len(s)), xerr=xerr, fmt="none", ecolor="#999999",
                        elinewidth=1.5, capsize=2, zorder=2)
            ax.scatter(s["theta_1sd_rel"], range(len(s)), c=colors, s=22, zorder=3, edgecolors="black", linewidths=0.4)
            ax.axvline(0, color="grey", lw=0.8, ls="--")
            ax.set_yticks(range(len(s)))
            ax.set_yticklabels(s["id"], fontsize=7)
            ax.set_title(f"{role}\n({palanca_nombre})", fontsize=9)
            ax.set_xlabel("theta_1sd_rel")
        fig.suptitle(f"ST-03 forest plot theta_1sd_rel por palanca -- {fase} (variante primitiva)")
        fig.tight_layout()
        fname = FIGS / f"st_03_forest_{'Reduccion' if fase == 'Reducción' else 'Fusion'}.png"
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        log(f"  {fname.name} guardado")

    for fase in ["Reducción", "Fusión"]:
        sub = cadena_df[(cadena_df["fase"] == fase) & (cadena_df["variante"] == "primitiva")]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="id", columns="role", values="valor_heatmap", aggfunc="first")
        piv = piv[[r for r in ROLES if r in piv.columns]]
        fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(piv))))
        im = ax.imshow(piv.values, cmap="coolwarm_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=7)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{int(v):+d}" if v != 0 else "0", ha="center", va="center", fontsize=7)
        ax.set_title(f"ST-03 cadena de signos -- {fase} (primitiva)\n+1 coherente / -1 incoherente / 0 indeterminado")
        fig.colorbar(im, ax=ax, shrink=0.6)
        fig.tight_layout()
        fname = FIGS / f"st_03_heatmap_{'Reduccion' if fase == 'Reducción' else 'Fusion'}.png"
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        log(f"  {fname.name} guardado")

    # =========================================================================
    # 7) Resumen final (honesto, con numeros) por fase
    # =========================================================================
    log("\n" + "=" * 90)
    log("RESUMEN FINAL POR FASE (variante primitiva salvo que se indique)")
    log("=" * 90)
    for fase in ["Reducción", "Fusión"]:
        log(f"\n--- {fase} ---")
        sub_theta = theta_df[(theta_df["fase"] == fase) & (theta_df["variante"] == "primitiva")]
        # (a) candidatos con mas palancas significativas y con signo teorico correcto
        concuerda_sig = sub_theta[sub_theta["significativo"] & (sub_theta["concuerda_teoria"] == True)]  # noqa: E712
        ranking_a = concuerda_sig.groupby("id").size().sort_values(ascending=False)
        log("(a) Nro de palancas significativas Y con signo teorico correcto, por candidato:")
        log(ranking_a.to_string() if len(ranking_a) else "  (ninguno)")

        # (b) cadena de signos coherente en carbon Y gn
        res_fase = resumen_df[(resumen_df["fase"] == fase) & (resumen_df["variante"] == "primitiva")]
        both_coherente = res_fase[(res_fase["coherente_carbon"] == "coherente") & (res_fase["coherente_gn"] == "coherente")]
        log("\n(b) Candidatos con cadena de signos COHERENTE en carbon Y gn (las 2 palancas con evidencia de batch mas fuerte):")
        if len(both_coherente):
            for id_ in both_coherente["id"]:
                fc = cadena_df[(cadena_df.id == id_) & (cadena_df.variante == "primitiva") & (cadena_df.role.isin(["carbon", "gn"]))]
                ft = theta_df[(theta_df.id == id_) & (theta_df.variante == "primitiva") & (theta_df.role.isin(["carbon", "gn"]))]
                log(f"  {id_}:")
                for role in ["carbon", "gn"]:
                    tr = ft[ft.role == role].iloc[0]
                    cr = fc[fc.role == role].iloc[0]
                    log(f"    {role:6s} theta_1sd_rel={tr.theta_1sd_rel:+.3f} IC1sd=[{tr.ci_lo_1sd:+.1f},{tr.ci_hi_1sd:+.1f}] "
                        f"sig={tr.significativo}  rho_target_rend={cr.rho_target_rendimiento:+.3f}  "
                        f"exp19({cr.exp19_variable})={cr.exp19_coef_por_sd:+.4f}(p={cr.exp19_p:.3f})")
        else:
            log("  (ninguno)")

        # (c) incoherentes
        incoherentes = res_fase[res_fase["n_incoherente"] > 0]
        log(f"\n(c) Candidatos con AL MENOS UNA palanca INCOHERENTE (maximizar el target empuja esa palanca en contra del KPI):")
        if len(incoherentes):
            for id_ in incoherentes["id"]:
                fc = cadena_df[(cadena_df.id == id_) & (cadena_df.variante == "primitiva") & (cadena_df.estado_cadena == "incoherente")]
                ft = theta_df[(theta_df.id == id_) & (theta_df.variante == "primitiva")]
                log(f"  {id_}:")
                for _, cr in fc.iterrows():
                    tr = ft[ft.role == cr.role].iloc[0]
                    log(f"    {cr.role:6s} theta_1sd_rel={tr.theta_1sd_rel:+.3f} IC1sd=[{tr.ci_lo_1sd:+.1f},{tr.ci_hi_1sd:+.1f}] "
                        f"rho_target_rend={cr.rho_target_rendimiento:+.3f}  exp19({cr.exp19_variable})={cr.exp19_coef_por_sd:+.4f}(p={cr.exp19_p:.3f})")
        else:
            log("  (ninguno)")

    log(f"\nTiempo total script: {(time.time()-t_ini)/60:.1f} min")
    log("ST-03 terminado.")
    _log_f.close()


if __name__ == "__main__":
    main()
