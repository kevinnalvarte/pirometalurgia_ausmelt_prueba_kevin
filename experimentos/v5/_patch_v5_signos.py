"""Parche de modelo_predictivo_v5: theta con signos de teoria (lsq acotado) y soporte relativo en J."""
from pathlib import Path
p = Path(__file__).resolve().parent.parent.parent / "modelo_predictivo_v5.py"
s = p.read_text(encoding="utf-8")

marker = "# =============================================================================\n# 2) Sistema v5"
plm_cls = '''# =============================================================================
# 1b) PLM con restricciones de signo por teoria (E7-03/E7-05)
# =============================================================================

class ModeloPLMSignos(mp4.ModeloPLM):
    """PLM v4 cuyos theta se estiman por minimos cuadrados ACOTADOS con el signo que dicta la teoria
    (SIGNO_TEORICO_V5: +1 -> theta >= 0, -1 -> theta <= 0, 0 -> libre). Es la unica monotonia impuesta
    en v5 y solo sobre el efecto lineal de las palancas de CONTROL: un reductor (carbon, GN) no puede
    aumentar la retencion de FeO ni reducir el agotamiento de Sn; una lanza mas oxidante (exceso de O2)
    no puede aumentar el agotamiento de Sn ni reducir la retencion de FeO. Si el dato contradice el signo
    (IC que cruza 0), el theta colapsa a 0 = "sin efecto identificado" en vez de un efecto espurio de signo
    fisicamente imposible (p. ej. carbon +0.0004 sobre ln_feo_ret, E7-05 provisional). El estado g(S)
    sigue sin restricciones. IC95 % por bootstrap cluster (misma restriccion en cada replica)."""

    def __init__(self, estado, palancas, target, signos: dict | None = None):
        super().__init__(estado, palancas, target)
        self.signos = dict(signos or {})

    def _bounds(self):
        lo = np.array([0.0 if self.signos.get(a, 0) > 0 else -np.inf for a in self.palancas])
        hi = np.array([0.0 if self.signos.get(a, 0) < 0 else np.inf for a in self.palancas])
        return lo, hi

    def _lstsq(self, A, y):
        from scipy.optimize import lsq_linear
        lo, hi = self._bounds()
        if np.all(np.isinf(lo)) and np.all(np.isinf(hi)):
            return np.linalg.lstsq(A, y, rcond=None)[0]
        return lsq_linear(A, y, bounds=(lo, hi), lsmr_tol="auto").x

    def fit(self, d: pd.DataFrame, n_splits: int = 5, n_boot: int = 200, seed: int = SEED):
        S, A, y, grupos = self.estado, self.palancas, self.target, d["Batch"].to_numpy()
        cols = [y] + A
        oof = pd.DataFrame(np.nan, index=d.index, columns=cols)
        for tr, te in GroupKFold(n_splits).split(d[S], d[y], grupos):
            for c in cols:
                mdl = mp4._hgb_nuisance(seed).fit(d[S].iloc[tr], d[c].iloc[tr])
                oof.iloc[te, oof.columns.get_loc(c)] = mdl.predict(d[S].iloc[te])
        y_res = d[y].to_numpy() - oof[y].to_numpy()
        A_res = np.column_stack([d[a].to_numpy() - oof[a].to_numpy() for a in A])
        self.theta = self._lstsq(A_res, y_res)
        self.res_oof = y_res - A_res @ self.theta
        self.r2_oof_interno = float(r2_score(d[y], d[y].to_numpy() - self.res_oof))
        rng = np.random.default_rng(seed)
        uniq = np.unique(grupos)
        idx_by = {b: np.where(grupos == b)[0] for b in uniq}
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
            boots.append(self._lstsq(A_res[idx], y_res[idx]))
        if boots:
            boots = np.array(boots)
            self.ci_lo, self.ci_hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        else:
            self.ci_lo, self.ci_hi = np.full(len(A), np.nan), np.full(len(A), np.nan)
        self.sd_palancas = d[A].std().to_numpy()
        self.g = mp4._hgb_nuisance(seed).fit(d[S], d[y])
        self.m = {a: mp4._hgb_nuisance(seed).fit(d[S], d[a]) for a in A}
        self.n_train = len(d)
        return self


def _nuevo_plm(fase: str, clave: str, S: list[str], A: list[str], target: str):
    return ModeloPLMSignos(S, A, target, SIGNO_TEORICO_V5.get((fase, clave)))


'''
assert marker in s
s = s.replace(marker, plm_cls + marker)
reps = [
("            sistema.modelos[(fase, clave)] = mp4.ModeloPLM(S, A, target).fit(d, n_boot=n_boot, seed=seed)",
 "            sistema.modelos[(fase, clave)] = _nuevo_plm(fase, clave, S, A, target).fit(d, n_boot=n_boot, seed=seed)"),
('''def _oof_plm(d_dev: pd.DataFrame, S: list[str], A: list[str], target: str, n_splits: int, seed: int) -> np.ndarray:
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(d_dev[S], d_dev[target], d_dev["Batch"]):
        m = mp4.ModeloPLM(S, A, target).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)''',
 '''def _oof_plm(d_dev: pd.DataFrame, S: list[str], A: list[str], target: str, n_splits: int, seed: int, signos: dict | None = None) -> np.ndarray:
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(d_dev[S], d_dev[target], d_dev["Batch"]):
        m = ModeloPLMSignos(S, A, target, signos).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)'''),
('''            oof = _oof_plm(d_dev, S, A, target, n_splits, seed)
            m = mp4.ModeloPLM(S, A, target).fit(d_dev, n_boot=0, seed=seed)
            p_lb = m.predict(d_lb)''',
 '''            oof = _oof_plm(d_dev, S, A, target, n_splits, seed, SIGNO_TEORICO_V5.get((fase, clave)))
            m = _nuevo_plm(fase, clave, S, A, target).fit(d_dev, n_boot=0, seed=seed)
            p_lb = m.predict(d_lb)'''),
('''    oof = _oof_plm(d_dev, S, A, target, n_splits, seed)
    m = mp4.ModeloPLM(S, A, target).fit(d_dev, n_boot=0, seed=seed)
    keep =''',
 '''    oof = _oof_plm(d_dev, S, A, target, n_splits, seed, SIGNO_TEORICO_V5.get((fase, clave)))
    m = _nuevo_plm(fase, clave, S, A, target).fit(d_dev, n_boot=0, seed=seed)
    keep ='''),
('''def objetivo_lote_v5(sistema: SistemaPrescriptivoV5, fase: str, state: dict, acciones: list[dict], context: dict | None = None,
                     pesos: dict | None = None) -> pd.DataFrame:''',
 '''def objetivo_lote_v5(sistema: SistemaPrescriptivoV5, fase: str, state: dict, acciones: list[dict], context: dict | None = None,
                     pesos: dict | None = None, soporte_ref: float | None = None) -> pd.DataFrame:'''),
('''    sim["J_soporte"] = -p["w_soporte"] * (1 - sim["support_score"])
    if fase == "Reducción":
        w = cfg["w_sdi"]''',
 '''    # Soporte RELATIVO (v5): solo se penaliza perder densidad historica respecto de la accion de referencia (la historica);
    # ganar densidad no se premia (en v4 el termino absoluto -w_S*(1-s) dominaba a la fisica y empujaba hacia la moda historica).
    if soporte_ref is None:
        sim["J_soporte"] = -p["w_soporte"] * (1 - sim["support_score"])
    else:
        sim["J_soporte"] = -p["w_soporte"] * np.maximum(0.0, soporte_ref - sim["support_score"])
    if fase == "Reducción":
        w = cfg["w_sdi"]'''),
('''    sim = objetivo_lote_v5(sistema, fase, state, candidatos, context, pesos)
    umbral = sistema.soporte_minimo.get(fase, 0.0)
    J_adm = sim["J"].where(sim["support_score"] >= umbral)''',
 '''    s_ref = float(simular_acciones_lote_v5(sistema, fase, state, [accion_base], context)["support_score"].iloc[0])
    sim = objetivo_lote_v5(sistema, fase, state, candidatos, context, pesos, soporte_ref=s_ref)
    umbral = sistema.soporte_minimo.get(fase, 0.0)
    J_adm = sim["J"].where(sim["support_score"] >= umbral)'''),
('''        sim_r = objetivo_lote_v5(sistema, fase, state, refin, context, pesos)
        J_r = sim_r["J"].where(sim_r["support_score"] >= umbral)''',
 '''        sim_r = objetivo_lote_v5(sistema, fase, state, refin, context, pesos, soporte_ref=s_ref)
        J_r = sim_r["J"].where(sim_r["support_score"] >= umbral)'''),
('''    sim = objetivo_lote_v5(sistema, fase, state, acciones, context, pesos)
    sim.insert(0, "valor", grid); sim.insert(0, "palanca", palanca)''',
 '''    s_ref = float(simular_acciones_lote_v5(sistema, fase, state, [action], context)["support_score"].iloc[0])
    sim = objetivo_lote_v5(sistema, fase, state, acciones, context, pesos, soporte_ref=s_ref)
    sim.insert(0, "valor", grid); sim.insert(0, "palanca", palanca)'''),
('''            sim_h = objetivo_lote_v5(sistema, fase, state, [accion_hist], context, pesos).iloc[0]''',
 '''            s_h = float(simular_acciones_lote_v5(sistema, fase, state, [accion_hist], context)["support_score"].iloc[0])
            sim_h = objetivo_lote_v5(sistema, fase, state, [accion_hist], context, pesos, soporte_ref=s_h).iloc[0]'''),
]
for a, b in reps:
    assert a in s, a[:60]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("patched", s.count("ModeloPLMSignos"))
