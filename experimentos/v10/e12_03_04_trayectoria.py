"""E12-03 tarea 5: trayectorias suavizadas (RTS, descriptivo) de beta_G y beta_C con bandas +-1.96 sd, junto a
T_media_R; y prueba simple de quiebre (sup-F / Chow en rejilla de tau) para xG y xC, con p-valor por permutacion
en bloques de 20 (misma nula que en el resto de E12-03/E11-07)."""
import sys

sys.path.insert(0, "experimentos/v9")
sys.path.insert(0, "experimentos/v10")
import numpy as np
import pandas as pd
import e11_07_lib as E
import e12_03_lib as K

t = E.cargar_tabla()
D = E.preparar(t, E.X_PRINCIPAL)
Z = K.construir_Z(D)
y, Xctrl, Xexp = D["y"], D["Xctrl"], D["Xexp"]
nc = Xctrl.shape[1]
n = D["n"]
idx = D["idx"]
T_R = D["T_media_R"]
es_dev = D["es_dev"]

# ------------------------------------------------------------------------------------------------- (a) suavizado
# NOTA: aqui (solo descriptivo) los hiperparametros se eligen por verosimilitud predictiva en TODA la muestra
# (no solo el burn-in [40,100)): con el burn-in, la rejilla selecciona q_G=q_C=0 (ver e12_03_01_filtro.py) porque
# el arranque de campana aun no muestra deriva detectable -> el suavizador colapsaria a una recta (beta constante
# = OLS pleno). Para DESCRIBIR la trayectoria se necesita una q representativa de la deriva ya documentada en
# E11-07 (beta_G: 0 -> -2.8 -> +0.5). Esta eleccion NO se usa en la prueba prospectiva (tareas 1-4, causales).
GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
beta0, P0 = K.init_ols(Z, y, K.NINIT)
r_full = E.ols_classic(Z[:, 1:], y)
R_full = float(r_full["resid"] @ r_full["resid"]) / max(len(y) - Z.shape[1], 1)
mejor = None
for qg in GRID:
    for qc in GRID:
        Qtest = [0.0] * (1 + nc) + [qg, qc]
        ll = K.loglik_predictiva(Z, y, Qtest, R_full, beta0, P0, K.NINIT, K.NINIT, n)
        if mejor is None or ll > mejor[0]:
            mejor = (ll, qg, qc)
print(f"hiperparametros del suavizador (verosimilitud en TODA la muestra, solo descriptivo): "
      f"q_G={mejor[1]}, q_C={mejor[2]}, R={R_full:.3f}")
Qd = np.array([0.0] * (1 + nc) + [mejor[1], mejor[2]])
res = K.kalman_run(Z, y, Qd, R_full, beta0=beta0, P0=P0, i0=K.NINIT)
sm = K.rts_smoother(res)
bG_s, bC_s = sm["beta_smooth"][:, -2], sm["beta_smooth"][:, -1]
sdG_s, sdC_s = np.sqrt(sm["P_smooth_diag"][:, -2]), np.sqrt(sm["P_smooth_diag"][:, -1])
bG_f, bC_f = res["beta_filt"][:, -2], res["beta_filt"][:, -1]

traj = pd.DataFrame({
    "idx_cronologico": idx, "es_dev": es_dev, "T_media_R": T_R,
    "beta_G_filt": bG_f, "beta_C_filt": bC_f,
    "beta_G_smooth": bG_s, "beta_C_smooth": bC_s,
    "sd_G_smooth": sdG_s, "sd_C_smooth": sdC_s,
    "beta_G_smooth_lo": bG_s - 1.96 * sdG_s, "beta_G_smooth_hi": bG_s + 1.96 * sdG_s,
    "beta_C_smooth_lo": bC_s - 1.96 * sdC_s, "beta_C_smooth_hi": bC_s + 1.96 * sdC_s,
}).iloc[K.NINIT:].reset_index(drop=True)
traj.to_csv(K.SALIDA / "e12_03_trayectoria.csv", index=False)
print(traj.iloc[::20].round(3).to_string())

# ------------------------------------------------------------------------------------------------- (b) sup-F / Chow
CTRL_COLS = 1 + nc  # columnas de Z antes de xG,xC (intercepto+controles)
Zctrl_exp = Z  # [1, controles, xG, xC]


def ssr(Zmat, yy):
    beta, *_ = np.linalg.lstsq(Zmat, yy, rcond=None)
    r = yy - Zmat @ beta
    return float(r @ r)


def sup_f(Xexp_local: np.ndarray, tau_grid: np.ndarray) -> tuple[float, int, np.ndarray]:
    Zr = np.column_stack([np.ones(n), Xctrl, Xexp_local])  # restringido: sin interaccion
    ssr_r = ssr(Zr, y)
    k_r = Zr.shape[1]
    fs = np.empty(len(tau_grid))
    for j, tau in enumerate(tau_grid):
        post = (np.arange(n) >= tau).astype(float)
        Zu = np.column_stack([Zr, Xexp_local[:, 0] * post, Xexp_local[:, 1] * post])
        ssr_u = ssr(Zu, y)
        k_u = Zu.shape[1]
        q = k_u - k_r
        dof = n - k_u
        fs[j] = ((ssr_r - ssr_u) / q) / (ssr_u / dof)
    j_best = int(np.argmax(fs))
    return float(fs[j_best]), int(tau_grid[j_best]), fs


tau_grid = np.arange(60, n - 60, 5)
obs_supf, tau_star, fs_curve = sup_f(Xexp, tau_grid)
print(f"sup-F observado = {obs_supf:.3f} en tau*={tau_star} (batch idx {idx[tau_star]})")

bloques = E.bloques_cronologicos(n, 20)
rng = np.random.default_rng(K.SEED)
N_PERM_CHOW = 1500
null_supf = np.empty(N_PERM_CHOW)
for i in range(N_PERM_CHOW):
    Xp = E.permutar_bloques(Xexp, bloques, rng)
    null_supf[i], _, _ = sup_f(Xp, tau_grid)
p_chow = float((np.sum(null_supf >= obs_supf) + 1) / (N_PERM_CHOW + 1))
print(f"p (permutacion bloques20, {N_PERM_CHOW} rep.) = {p_chow:.4f}")

pd.DataFrame({"tau": tau_grid, "batch_idx": idx[tau_grid], "sup_f_local": fs_curve}).to_csv(
    K.SALIDA / "e12_03_chow_curva.csv", index=False)
pd.DataFrame([{"sup_f_obs": obs_supf, "tau_star": tau_star, "batch_idx_tau_star": idx[tau_star],
               "p_perm_bloques20": p_chow, "n_perm": N_PERM_CHOW}]).to_csv(K.SALIDA / "e12_03_chow.csv", index=False)
