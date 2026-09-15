// ╔══════════════════════════════════════════════════════════════════════╗
// ║                        ══ COPYRIGHT NOTICE ══                        ║
// ║                    Code Master: Yair Camborda                        ║
// ║                   All Rights Reserved © Minsur SA                    ║
// ║                           Year 2024                                  ║
// ║----------------------------------------------------------------------║
// ║ Welcome to this flawless piece of art, also known as code. You are   ║
// ║ about to experience a program so refined, it's almost like it was    ║
// ║ coded by someone competent!                                          ║
// ║----------------------------------------------------------------------║
// ║ In the unforgiving wilderness of Rust, we handle memory leaks like   ║
// ║ a Jedi manages their emotions: We pretend they don't exist until     ║
// ║ they absolutely do. "May the Rust be with you"—or else.              ║
// ║----------------------------------------------------------------------║
// ║ Crafted with the precision of a Half-Life physics puzzle, this code  ║
// ║ runs tighter than a Naruto filler episode. Expect speeds so fast,    ║
// ║ Hollow Knight himself would want to dodge them.                      ║
// ║----------------------------------------------------------------------║
// ║ Here's the twist: despite its epic backstory of being resurrected    ║
// ║ from a previous disaster, this project is as stable as a D&D         ║
// ║ campaign run by a first-time dungeon master. Fear not, it’s          ║
// ║ definitely not going to crash and burn... probably.                  ║
// ║----------------------------------------------------------------------║
// ║ So, gear up, fellow coder. If this codebase were any more robust,    ║
// ║ it'd be a Dark Souls boss—unforgiving, mysterious, and rewarding     ║
// ║ only to the persistent.                                              ║
// ║----------------------------------------------------------------------║
// ║ Embrace the challenge, enjoy the bugs. May your coffee be strong     ║
// ║ and your compile times short.                                        ║
// ║                                                                      ║
// ║                      ~ Yair Camborda                                 ║
// ║                                                                      ║
// ║ PS: Remember, if something breaks, it's definitely by design.        ║
// ║ Just a feature waiting to be understood by mere mortals.             ║
// ╚══════════════════════════════════════════════════════════════════════╝

use crate::Number;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr, IfIsHumanReadable};
use std::collections::HashMap;

#[serde_as]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TheoreticalInput {
    pub batch_id: String,
    pub cama_id: String,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub num_batches: i32,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub tiempo_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub temperatura_inicial_escoria_red: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub temperatura_metal_fus: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub enriquecimiento_o2_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub estequiometria_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub nivel_escoria_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub aire_post_combustion_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub aire_infiltracion_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub carbon_perdido_tiro_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub temperatura_inicial_fusion_horno: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub temperatura_final_fusion_horno: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub temperatura_salida_gas_cooler_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub perc_escoria_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    factor_eficiencia_ca_o_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub factor_dross_fusion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub tiempo_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub enriquecimiento_o2_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub estequiometria_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub nivel_escoria_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub aire_post_combustion_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub aire_infiltracion_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub carbon_perdido_tiro_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub temperatura_inicial_reduccion_horno: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub temperatura_final_reduccion_horno: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub temperatura_salida_gas_cooler_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub perc_escoria_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub factor_eficiencia_ca_o_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub factor_dross_reduccion: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub velocidad_opt: i8,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub velocidad_opt_value: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub carbon_peletizado: i8,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub carbon_peletizado_value: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub densidad_escoria: Number,
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub diametro_interno: Number,
    pub recirculantes: HashMap<String, [Number; 2]>,
    pub otras_configuraciones: HashMap<String, Number>,
    pub dataframes: HashMap<String, String>,
    pub cama_dataframe: String,
    #[serde(flatten)]
    additional_info: HashMap<String, String>,
}

// Definir el HashMap usando lazy_static
lazy_static! {
    static ref MOLECULAR_WEIGHTS: HashMap<&'static str, Number> = {
        // TODO: REVIEW DECIMALS OF MASSES
        let mut m = HashMap::new();
        m.insert("c", 12.00);
        m.insert("o", 16.00);
        m.insert("s", 32.07);
        m.insert("pb", 207.20);
        m.insert("sn", 118.71);
        m.insert("sb", 121.76);
        m.insert("as", 74.92);
        m.insert("si", 28.09);
        m.insert("ca", 40.08);
        m.insert("al", 26.98);
        m.insert("fe", 55.85);
        m.insert("al2o3", 101.96);
        m.insert("cu", 63.55);
        m.insert("cao", 56.08);
        m.insert("sno2", 150.71);
        m.insert("fe2o3", 159.70);
        m.insert("fes2", 119.99);
        m.insert("sio2", 60.09);
        m.insert("sno", 134.71);
        m.insert("sns", 150.78);
        m.insert("pbo", 223.20);
        m.insert("cuo", 79.55);
        m.insert("sb2o3", 291.52);
        m.insert("as2o3", 197.85);
        m.insert("mg", 24.31);
        m.insert("mgo", 40.31);
        m.insert("caco3", 100.09);
        m.insert("mgco3", 84.32);
        m.insert("fesn2", 293.27);
        m.insert("cu2s", 159.17);
        m.insert("feo", 71.85);
        m.insert("zno", 81.37);
        m.insert("zn", 65.37);
        m.insert("co", 28.00);
        m.insert("s2", 64.14);
        m.insert("co2", 44.00);
        m.insert("c2h6", 30.00);
        m.insert("h", 1.00);
        m.insert("o2", 32.00);
        m.insert("h2o", 18.00);
        m.insert("so2", 64.07);
        m.insert("2caosio2", 172.25);
        m.insert("2feosio2", 203.79);
        m.insert("mgoal2o3", 142.27);
        m.insert("fe2", 2.0 * 55.85);
        m.insert("0.5s2", 32.07);
        m.insert("2sn", 2.0 * 118.71);
        m.insert("0.5o2", 16.00);
        m.insert("sno(v)", 134.71);
        m.insert("sno(esc)", 134.71);
        m.insert("2feo", 2. * 71.85);
        m.insert("feo(esc)", 71.85);
        m.insert("2cu", 2. * 63.55);
        m.insert("3h2o", 3. * 18.00);
        m.insert("3.5o2", 3.5 * 32.00);
        m.insert("2co2", 2. * 44.00);
        m.insert("2o2", 2. * 32.00);
        m
    };
}

// Función para acceder y usar el HashMap
pub fn mw(molecule: &str) -> Number {
    *MOLECULAR_WEIGHTS
        .get(molecule)
        .unwrap_or_else(|| panic!("*** {molecule} Not found ****"))
}
