"""E11-07 tarea 5: alternativas de re-anclaje (fijadas a priori), W base = 100 / equivalente, paso=10, muestra 'todo'.
(i) ponderacion exponencial (semivida 40/60/80) en vez de ventana dura.
(ii) ridge hacia 0 sobre los coeficientes de exposicion (gl efectivos ~= 1.5).
(iii) contraccion hacia el beta de largo plazo (media beta_ventana100 y beta_expansivo).
(iv) solo xG; solo xC.
(v) agrega exposiciones de Fusion x_F_carbon, x_F_exo2 (4 exposiciones)."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as E
import v9_lib as L

t = E.cargar_tabla(extra_cols=["x_F_carbon", "x_F_exo2"])
D = E.preparar(t, E.X_PRINCIPAL)
Xexp, Xctrl, y, n = D["Xexp"], D["Xctrl"], D["y"], D["n"]
mask = np.ones(n, bool)


def eval_score(score, yy=None, Xc=None, nombre=""):
    r = E.stats_final(score, y if yy is None else yy, Xctrl if Xc is None else Xc, np.ones(len(score), bool))
    r["variante"] = nombre
    return r

filas = [eval_score(E.walk_forward_score(Xexp, Xctrl, y, 100, paso=E.PASO), nombre="referencia W=100 duro")]

for hl in (40, 60, 80):
    sc = E.walk_forward_score(Xexp, Xctrl, y, None, paso=E.PASO, start=100, halflife=hl)
    filas.append(eval_score(sc, nombre=f"(i) ponderacion exponencial semivida={hl}"))

sc = E.walk_forward_score(Xexp, Xctrl, y, 100, paso=E.PASO, ridge_lambda=1.5)
filas.append(eval_score(sc, nombre="(ii) ridge hacia 0, gl_efectivos~=1.5 (ventana 100)"))

sc = E.walk_forward_score(Xexp, Xctrl, y, 100, paso=E.PASO, shrink_expansiva=True)
filas.append(eval_score(sc, nombre="(iii) contraccion: media(beta_ventana100, beta_expansivo)"))

sc = E.walk_forward_score(Xexp[:, [0]], Xctrl, y, 100, paso=E.PASO)
filas.append(eval_score(sc, nombre="(iv) solo xG"))
sc = E.walk_forward_score(Xexp[:, [1]], Xctrl, y, 100, paso=E.PASO)
filas.append(eval_score(sc, nombre="(iv) solo xC"))

D4 = E.preparar(t, E.X_PRINCIPAL + ["x_F_carbon", "x_F_exo2"])
sc4 = E.walk_forward_score(D4["Xexp"], D4["Xctrl"], D4["y"], 100, paso=E.PASO)
filas.append(eval_score(sc4, yy=D4["y"], Xc=D4["Xctrl"], nombre="(v) + x_F_carbon, x_F_exo2 (4 exposiciones)"))

out = pd.DataFrame(filas)[["variante", "n", "pendiente", "se", "t", "p_uni", "spearman", "sd_score"]]
out.to_csv(E.SALIDA / "e11_07_alternativas.csv", index=False)
pd.set_option("display.width", 220); print(out.round(4).to_string())
