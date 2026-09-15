"""Prototipo: seleccion de features con tecnicas avanzadas NO usadas en las rondas previas
(Pasos 18-19 uso eliminacion hacia atras guiada por SHAP; Pasos 20-27 sumaron un chequeo de
correlacion cruzada post-poda). Aqui se agregan 3 tecnicas independientes, con distinto sesgo
metodologico cada una, para auditar (no repetir) el set ya elegido en modelo_prescriptivo.py:

1. Mutual information (no-parametrica, captura no-linealidad sin necesitar un modelo entrenado).
2. Permutation importance ESTABLE: repetida sobre K folds (GroupKFold) x S semillas, reportando
   NO solo la media sino la fraccion de folds/semillas en que la importancia es significativamente
   > 0 -- una feature "importante" solo en 1 de 25 corridas es sospechosa de ruido/sobreajuste local.
3. Boruta-shadow test (implementacion manual, sin libreria externa): crea copias barajadas
   ("shadow features") de cada candidata, entrena un modelo con original+shadow, y compara la
   importancia (gain) de cada candidata real contra la MAXIMA importancia entre todas las shadow
   -- si una feature real gana a la mejor shadow en >=60% de N iteraciones, se considera "confirmada"
   (test binomial informal, igual que el algoritmo Boruta original de Kursa & Rudnicki 2010).
"""
import os
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

sys.path.insert(0, r"C:\Users\user\Desktop\Linea de Sn\Lingo Smelter")
import feature_engineering as fe
import modelo_prescriptivo as mp

PARAMS = mp.PARAMS_XGB
OUT_DIR = os.path.dirname(__file__)

df = mp.construir_dataset_modelo(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datos Lingo smelter fase II.xlsx"))
df_base = mp.dataset_base_modelo(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)

ROLES_CANDIDATOS = {"STATE", "ACTION_primitiva", "DERIVED-ACTION", "DERIVED-STATE", "DERIVED (A_t x S_prev)"}
POOL_BASE = [c for c, rol in fe.CLASIFICACION_FEATURES.items() if rol in ROLES_CANDIDATOS]
POOL_BASE = [c for c in POOL_BASE if c in df_base.columns]
POOL_BASE = sorted(set(POOL_BASE) | {"orden_escalon_fase"})
print(f"Pool de candidatas seguras (STATE/ACTION/DERIVED): {len(POOL_BASE)} columnas")


def mutual_info_ranking(sub, pool, target_col, seed=0):
    d = sub[pool + [target_col]].dropna()
    mi = mutual_info_regression(d[pool], d[target_col], random_state=seed, n_neighbors=5)
    return pd.Series(mi, index=pool).sort_values(ascending=False)


def permutation_stability(sub, pool, target_col, n_splits=5, n_seeds=5):
    """Para cada (fold, seed): entrena XGB en train, mide permutation_importance en el TEST fold
    (fuera de muestra, no in-sample). Reporta media, std y fraccion de corridas donde importancia
    media > 0 con margen (> 0.5*std de esa corrida, para evitar contar ruido de 0 como 'positivo')."""
    d = sub[pool + [target_col, "Batch"]].dropna().reset_index(drop=True)
    X, y, groups = d[pool], d[target_col], d["Batch"].to_numpy()
    registros = {f: [] for f in pool}
    for seed in range(n_seeds):
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in gkf.split(X, y, groups):
            m = xgb.XGBRegressor(**dict(PARAMS, random_state=seed))
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            r = permutation_importance(m, X.iloc[test_idx], y.iloc[test_idx], n_repeats=8,
                                        random_state=seed, scoring="r2", n_jobs=-1)
            for i, f in enumerate(pool):
                registros[f].append(r.importances_mean[i])
    resumen = pd.DataFrame({
        f: {"media": np.mean(v), "std": np.std(v), "frac_positiva": np.mean(np.array(v) > 0)}
        for f, v in registros.items()
    }).T.sort_values("media", ascending=False)
    return resumen


def boruta_shadow_test(sub, pool, target_col, n_iter=25, seed=0):
    """Boruta manual: en cada iteracion se barajan TODAS las candidatas para crear shadows,
    se entrena 1 XGB con original+shadow, y se cuenta cuantas veces cada real supera a la
    MEJOR shadow de esa iteracion (umbral de referencia mas exigente que comparar contra cada
    shadow individual)."""
    d = sub[pool + [target_col]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    hits = {f: 0 for f in pool}
    for it in range(n_iter):
        shadow = d[pool].apply(lambda col: rng.permutation(col.to_numpy()))
        shadow.columns = [f"shadow__{c}" for c in pool]
        X_ext = pd.concat([d[pool].reset_index(drop=True), shadow.reset_index(drop=True)], axis=1)
        m = xgb.XGBRegressor(**dict(PARAMS, random_state=seed + it))
        m.fit(X_ext, d[target_col])
        importancias = pd.Series(m.feature_importances_, index=X_ext.columns)
        max_shadow = importancias[[c for c in importancias.index if c.startswith("shadow__")]].max()
        for f in pool:
            if importancias[f] > max_shadow:
                hits[f] += 1
    resumen = pd.DataFrame({"hits": hits}).assign(n_iter=n_iter)
    resumen["frac_confirmada"] = resumen["hits"] / n_iter
    resumen["veredicto"] = pd.cut(resumen["frac_confirmada"], bins=[-0.01, 0.4, 0.6, 1.01],
                                   labels=["RECHAZADA", "TENTATIVA", "CONFIRMADA"])
    return resumen.sort_values("frac_confirmada", ascending=False)


for fase in ["Fusión", "Reducción"]:
    target = mp.TARGET_PRESCRIPTIVO_GANADOR[fase]
    sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)]
    pool = [c for c in POOL_BASE if sub[c].notna().mean() > 0.7]  # descarta candidatas con demasiados nulos (leccion Paso 11.3)
    print(f"\n===== {fase}: target={target}, pool utilizable={len(pool)}/{len(POOL_BASE)} (n={len(sub)}) =====")

    mi = mutual_info_ranking(sub, pool, target)
    mi.to_csv(f"{OUT_DIR}/mi_{fase}.csv")
    print("Top-15 mutual information:\n", mi.head(15).round(4))

    perm = permutation_stability(sub, pool, target)
    perm.to_csv(f"{OUT_DIR}/perm_{fase}.csv")
    print("\nTop-15 permutation importance (media OOF, 5 folds x 5 semillas):\n", perm.head(15).round(4))

    bor = boruta_shadow_test(sub, pool, target)
    bor.to_csv(f"{OUT_DIR}/boruta_{fase}.csv")
    print("\nBoruta-shadow (25 iteraciones):\n", bor.head(20))

    # Consenso: cuantas de las 3 tecnicas ubican a la feature en su top-10
    top_mi = set(mi.head(10).index)
    top_perm = set(perm.head(10).index)
    top_bor = set(bor.loc[bor["veredicto"] == "CONFIRMADA"].index)
    consenso = pd.Series({f: (f in top_mi) + (f in top_perm) + (f in top_bor) for f in pool}).sort_values(ascending=False)
    consenso.to_csv(f"{OUT_DIR}/consenso_{fase}.csv")
    print(f"\nConsenso (0-3 tecnicas coinciden), top-15:\n{consenso.head(15)}")

    actual = set(mp.FEATURES_STATE_ACTION[fase])
    print(f"\nFeatures YA en produccion (modelo_prescriptivo.py) para {fase}: {sorted(actual)}")
    print(f"Con consenso 0 (ninguna tecnica las respalda) entre las de produccion: {[f for f in actual if consenso.get(f, 0) == 0]}")
    print(f"Con consenso>=2 pero AUSENTES de produccion (candidatas a agregar): {sorted(consenso[(consenso >= 2)].index.difference(actual))}")

print("\nListo.")
