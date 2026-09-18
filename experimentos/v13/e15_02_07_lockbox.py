"""E15-02 punto 7: por que el lockbox da ~=0? Trayectoria de beta_d_gn/beta_d_carbon (OLS crudo, ventana
dura W=100, paso 10) a lo largo de toda la corrida, y su t. Coeficientes por regimen (terciles de DEV +
lockbox) con OLS estatico. Se chequea si el asesor se "abstendria" (|t|<1, como v9) justo en las ventanas
que puntuan los batches de lockbox."""
import sys, time
sys.path.insert(0, "experimentos/v13")
import numpy as np, pandas as pd
import e15_02_lib as E
import e11_07_lib as W7
import v9_lib as L

t0 = time.time()
r, t = E.cargar()
P = W7.preparar(t, ["d_gn", "d_carbon"])
X, C, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]
idx = P["idx"]  # Batch labels en el orden usado
es_dev = P["es_dev"]

W, paso = 100, 10
filas = []
s = W
while s < n:
    lo = s - W
    Z = np.column_stack([np.ones(s - lo), C[lo:s], X[lo:s]])
    r_ = W7.ols_classic(np.column_stack([C[lo:s], X[lo:s]]), y[lo:s])
    ne = X.shape[1]
    b, se = r_["beta"][-ne:], r_["se"][-ne:]
    tval = np.divide(b, se, out=np.full_like(b, np.nan), where=se > 0)
    aplica_dev = bool(es_dev[s]) if s < n else None
    filas.append({"s_ini_ventana": lo, "s_fin_ventana": s, "aplica_a": "DEV" if aplica_dev else "LB",
                  "b_gn": b[0], "se_gn": se[0], "t_gn": tval[0], "b_carbon": b[1], "se_carbon": se[1], "t_carbon": tval[1]})
    s += paso
out = pd.DataFrame(filas)
out.to_csv("experimentos/v13/e15_02_07_trayectoria_beta.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(3).to_string())

n_abstendria_gn = int((out["t_gn"].abs() < 1).sum())
lb_rows = out[out.aplica_a == "LB"]
print(f"\nventanas totales: {len(out)} | con |t_gn|<1 (abstencion): {n_abstendria_gn}")
print(f"ventanas que puntuan LOCKBOX: {len(lb_rows)} | de esas con |t_gn|<1: {int((lb_rows.t_gn.abs() < 1).sum())} "
      f"| con |t_carbon|<1: {int((lb_rows.t_carbon.abs() < 1).sum())}")
print("\nultimas 6 ventanas (las que puntuan el tramo final):\n", out.tail(6).round(3).to_string())

# --- coeficientes por regimen: terciles de DEV + lockbox, OLS estatico (HC3)
tt = t.sort_values("idx_cronologico").copy()
dev_pos = np.where(tt.es_dev.to_numpy())[0]
terc_de_dev = pd.qcut(np.arange(len(dev_pos)), 3, labels=["DEV_t1", "DEV_t2", "DEV_t3"])
tt["regimen"] = "LOCKBOX"
tt.iloc[dev_pos, tt.columns.get_loc("regimen")] = terc_de_dev.astype(str)

filas_r = []
for reg, sub in tt.groupby("regimen"):
    cols = ["d_gn", "d_carbon"] + E.CTRL + [E.KPI]
    dd = sub.dropna(subset=cols)
    if len(dd) < 15:
        continue
    f = L.ols_hc3(dd[E.KPI], dd[["d_gn", "d_carbon"] + E.CTRL])  # KPI ya esta en puntos porcentuales
    filas_r.append({"regimen": reg, "n": len(dd), "b_gn": f.params["d_gn"], "t_gn": f.tvalues["d_gn"],
                     "b_carbon": f.params["d_carbon"], "t_carbon": f.tvalues["d_carbon"]})
out_r = pd.DataFrame(filas_r).sort_values("regimen")
out_r.to_csv("experimentos/v13/e15_02_07_regimenes.csv", index=False)
print("\ncoeficientes por regimen (OLS HC3):\n", out_r.round(3).to_string())
print(f"[{time.time()-t0:.0f}s]")
