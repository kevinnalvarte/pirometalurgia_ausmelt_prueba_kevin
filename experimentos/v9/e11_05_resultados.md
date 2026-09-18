# E11-05 -- vinculo directo de los controles de Fusion con el KPI del batch (H5)

Scripts `e11_05_a_descriptivo.py` (1), `e11_05_b_regresion_conjunta.py` (2), `e11_05_c_subgrupos_moderacion.py` (3, 7),
`e11_05_d_mecanismo.py` (4), `e11_05_e_placebos_anidada.py` (5, 6). CSV con el mismo prefijo. DEV n 297-299 / lockbox 63 / total 360-362
segun disponibilidad de controles.

## 1. Auditoria de `exo2`: no es una deriva de campana

| palanca | ICC (entre/intra batch) | autocorr(1) nivel | corr(idx_crono) nivel | autocorr(1) residual z | corr(idx_crono) residual z |
|---|---|---|---|---|---|
| exo2 | 0.13 | 0.07 | 0.03 | 0.07 | -0.02 |
| O2 | 0.66 | 0.76 | 0.68 | 0.18 | 0.18 |
| aire | 0.28 | 0.16 | -0.39 | -0.01 | -0.00 |
| GN | 0.57 | 0.60 | 0.58 | 0.04 | 0.16 |
| carbon | 0.14 | 0.45 | 0.18 | 0.58 | 0.31 |

O2 y GN (nivel bruto) SI son casi una tendencia de campana (ICC y autocorrelacion altas, coherente con el ajuste del proceso a lo
largo de la campana); el residuo `z` (lo que entra a las regresiones, tras HGB sobre el estado) absorbe la mayor parte de esa deriva.
`exo2`, en cambio, no tiene estructura temporal ni en nivel ni en residuo: ICC bajo, autocorrelacion ~0.07, correlacion con el reloj
~0. El R2 bajo de E[exo2\|S] (~0.08, iteracion previa) es ruido/variabilidad escalon a escalon (o fragilidad del indice, que es un
cociente con denominador 2*GN cerca de 0 en algunos escalones: rango observado -91 a +57), **no confusion temporal**.

Robustez del beta exo2->KPI (DEV) a controles temporales flexibles: base -1.00 (p .055) -> +poly3 idx -1.07 (p .046) -> +FE bloques
de 30 -0.89 (p .095) -> exposicion des-tendenciada (residuo vs media movil +-10 batches) -0.75 (p .19). El efecto no se explica por
una tendencia temporal (consistente con el punto anterior) pero se atenua y pierde significancia al des-tendenciar, señal de que
parte de la varianza usada es de muy corto plazo (ruido).

## 2. Regresion conjunta (OLS HC3, + exposiciones de Reduccion x_Rt/x_Rl como control)

DEV (n 297): `exo2` -> KPI **-1.01 pp/sd (p .067, q-BH .251)**; f_metal -1.10 (p .048, q .242); dross +1.26 (p .044, q .242); carbon_F
-> polvo +1.04 (p .007, q .103, borderline). **Ningun vinculo de Fusion pasa el criterio 1 (p<.05 Y q<.10) en DEV.**
Total (n 360, incluye lockbox "gastado"): exo2->KPI -1.42 (p .001, q .007); f_metal -1.39 (p .0004, q .006); dross +1.15 (p .014,
q .051); carbon_F->polvo +0.83 (p .006, q .029) / dross -0.94 (p .018, q .052). Lockbox solo (n 63): mismo signo, no significativo
(exo2->KPI -2.00, p .31).

Decomposicion del primitivo (spec B, O2 y aire por separado en vez de exo2): **O2** lleva el efecto (DEV -1.69, p .022; total -1.80,
p .002); **aire** no (~0 en todos los regimenes, p >= .38). GN y O2 estan colineales (r .83): al sacar exo2 del modelo, GN cambia de
signo (DEV +1.23 n.s., total +1.67 p .02) -- **no se puede separar limpiamente O2 de GN con estos datos**.

Tercios cronologicos (spec exo2->KPI): T1 -0.71 (p .55), T2 -0.60 (p .55), T3 -1.96 (p .015) -- **mismo signo en 3/3** pero
significativo solo en el ultimo tercio (fin de campana).

Bootstrap por batch (1000, DEV): exo2->KPI IC95 **[-2.37, +0.19]** (incluye 0); exo2->dross IC95 [0.23, 2.69] (excluye 0, el canal
mas robusto); carbon_F->polvo IC95 [0.32, 1.75] (excluye 0); carbon_F->dross IC95 [-1.35, 0.31] (incluye 0, mas debil que en E10-06
sin controlar por Reduccion).

## 3. Subgrupos F1-F3 / F4-F6 y moderacion

El par carbon_F se separa por sub-ventana: **F4-F6** carbon -> dross -0.98 (total, p .0004) / -0.62 (DEV, p .072); **F1-F3** carbon ->
polvo +0.67 (DEV, p .013) / +0.48 (total, p .041). Dos mecanismos distintos en dos mitades de Fusion, pero el neto sobre KPI sigue
sin diferir de 0 en ninguna mitad (todas p > .10). `exo2` -> KPI se concentra en F1-F3 (DEV -0.75 p .18, total -0.74 p .083) vs
F4-F6 (~0, n.s.), coherente con "lanza oxidante temprana" pero sin significancia propia.

Moderacion por estado previo (T, %Sn, %FeO al inicio de F1): **ninguna interaccion significativa** (p entre .24 y .92, exo2 y
carbon_F) -- a diferencia del GN de Reduccion (H2), el valor de la lanza de Fusion no depende medible-mente del estado de entrada.

Carbon_F neto por regimen (ley Sn conc. alta/baja, T media F alta/baja): todos n.s. (p .14-.90) -- confirma E10-06 (neto ~ 0) en
todos los regimenes examinados, sin un nicho donde el carbon de Fusion sea rentable.

## 4. Mecanismo de escalon (PLM residual-sobre-residual, DEV, GroupKFold5 sobre `ESTADO_F`)

R2 OOF de E[target\|estado]: `m6_ln_sn_dep` 0.13, `v5_d_lnirf` 0.44, `d_temperatura_horno_celsius` 0.38 (targets bien explicados por
el estado). `z__exo2` (residuo de la lanza) **no mueve ninguno**: sn_dep p .59, d_lnirf p .12, dT p .80. `z__o2` sí roza: sn_dep p
.098, dT **-1.23 C/sd (p .018)**, coherente en signo con el theta libre de E10-06 (O2 enfria). `z__carbon`: d_lnirf +0.0045/sd (p
.012), muy debil. **La cadena teorica (lanza -> mas Fe3+/magnetita -> mas Sn oxidado) no se sostiene al nivel del escalon** para
exo2.

Estado de entrada a Reduccion (T, %Sn, %FeO, FeO_inv al cierre de F6, via el estado de R0): `x_F_exo2` no predice nada (p > .23 en
las 4 variables). `x_F_carbon` sí: **baja el FeO que entra a Reduccion** (-325 kg DEV p .008 / -265 kg total p .015; %FeO -0.40 pp
DEV p .006 / -0.38 total p .002) y sube ligeramente el %Sn de entrada (total +0.91 pp, p .016). Mecanismo coherente (el carbon
desoxida FeO en Fusion) pero que no se traduce en ganancia neta de KPI (§2-3), quiza porque Reduccion se adapta al inventario que
recibe.

## 5. Placebos

(a) Exposicion de Fusion del batch SIGUIENTE -> KPI actual: no significativo (exo2_next DEV p .21 / total p .96; carbon_F_next DEV
p .99 / total p .12). (b) Exposicion -> pre-tratamiento: `x_F_exo2` sin senal en DEV (ley Sn p .27, feed Sn p .18, IRF inicial p
.32, KPI previo p .19); en total aparecen señales debiles (ley Sn p .038, IRF inicial p .0995) que no replican en DEV. `x_F_carbon`
se asocia fuerte con la ley de Sn del concentrado (DEV p .0004, total p<.0001) -- esperable (dosis de carbon escalada al Sn/receta)
y ya controlado en todas las regresiones (`ley_sn_conc_batch_pct` esta en `CONTROLES`).

## 6. Validacion anidada (`L.prueba_anidada`, DEV, gkf x20 / crono)

| especificacion | gkf pendiente (p_perm) | crono pendiente (p_perm) |
|---|---|---|
| solo x_F_exo2 | 0.71 (.123) | 0.47 (.263) |
| x_F_exo2 + x_F_carbon | 0.49 (.170) | 0.62 (.139) |
| + Reduccion (Rt/Rl carbon y gn) | 0.54 (**.0195**) | 0.61 (**.0095**) |
| solo Reduccion (sin Fusion) | 0.56 (**.038**) | 0.61 (**.016**) |

`exo2` solo o con carbon_F **no predice el KPI OOF** (p_perm > .12 en ambos esquemas). Solo pasa el criterio 4 cuando se agrega
Reduccion -- pero Reduccion **sola** ya pasa (t 1.74/1.98 vs 2.05/2.33 con Fusion). El aporte incremental de Fusion a la
demostracion obligatoria es marginal (2 parametros libres mas, sin poder OOF propio).

## Veredicto (criterios 1, 4, 6 del diseno)

- **Criterio 1**: falla en DEV (p .067 > .05; q-BH .251 > .10 para exo2->KPI). Solo se cumple en la regresion "total" (regimen no
  confirmatorio, incluye el lockbox ya "gastado").
- Canal: dross/f_metal son coherentes con la teoria y el bootstrap los sostiene (IC excluye 0), pero el IC de KPI mismo incluye 0.
- Tercios: mismo signo 3/3, pero el efecto esta **concentrado en el tercio final** (fin de campana) -- aunque §1 descarta una deriva
  temporal suave, no se descarta un cambio de regimen de fin de campana no capturado por los controles.
- Mecanismo de escalon (§4): **no soportado** -- exo2 no mueve Sn/IRF/dT dentro de Fusion ni predice el estado de entrada a
  Reduccion. La asociacion a nivel de batch no tiene una via fisica demostrable con estos datos.
- Criterio 4 (demostracion OOF obligatoria): **falla para Fusion sola**; el aporte incremental sobre Reduccion es marginal.
- Criterio 6 (parsimonia): **falla** -- el IC bootstrap de exo2->KPI incluye 0 y no hay moderacion por estado que lo sostenga.

**Fusion sigue en "receta".** No hay un vinculo control->KPI de Fusion que sobreviva los 6 criterios simultaneamente. La pista de
`exo2` (y de O2 en particular, no aire) es la mas prometedora de las dos hipotesis de Fusion pero no alcanza el umbral
pre-registrado: es sugestiva (mismo signo siempre, canal dross robusto, no es un artefacto de tendencia de campana) mas no
defendible como recomendacion con la evidencia y los criterios fijados. El carbon de Fusion mantiene su lectura de E10-06 (polvo
arriba / dross abajo, neto ~0 en todo regimen), con el matiz nuevo de que sí baja medible-mente el FeO que hereda Reduccion sin que
eso se traduzca en KPI.
