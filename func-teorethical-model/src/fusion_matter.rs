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

use crate::helpers::{DataFrameCustomExtensions, OptimizationHelpers};
use crate::json_manager::JSONManager;
use crate::{HmHmDf, Number};
use polars::prelude::col;
use polars::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Formatter};

pub struct FusionMatter {
    pub df_income_charge: DataFrame,
    pub df_combustion_income: DataFrame,
    pub reactions: Vec<HashMap<String, Number>>,
    pub df_dross_fe: DataFrame,
    pub df_metal: DataFrame,
    pub df_fumes: DataFrame,
    pub hm_post_combustion_fumes: HashMap<String, Number>,
    pub df_dump: DataFrame,
    pub hm_second_dump: HashMap<String, Number>,
    pub hm_third_dump: HashMap<String, Number>,
    pub hm_gases: HashMap<String, Number>,
    pub hm_gases_post: HashMap<String, Number>,
    pub df_gases_chimney: DataFrame,
    pub hm_balance_elements: HashMap<String, Number>,
}

impl OptimizationHelpers for FusionMatter {}

impl FusionMatter {
    pub fn new(json_manager: &mut JSONManager) -> Result<Self, Box<dyn Error>> {
        // Self::debug_message("FUSION CREATION START".to_string());
        let df_income_charge = Self::create_df_income_charge(json_manager)?;
        let (df_combustion_income, air_hashmap) =
            Self::create_df_combustion_income(json_manager, &df_income_charge)?;

        let hashmap_charge = df_income_charge.generate_hashmap_str_number("molecules", "total")?;

        let total_dist_tm_fe = hashmap_charge["fesn2"] * 55.85 / (55.85 + 2. * 118.7)
            + hashmap_charge["fe2o3"] * (2. * 55.85) / (2. * 55.85 + 3. * 16.)
            + hashmap_charge["fes2"] * 55.85 / (55.85 + 2. * 32.)
            + hashmap_charge["feo"] * 55.85 / (55.85 + 16.)
            + hashmap_charge["fe"];
        json_manager.total_dist_tm_fe = total_dist_tm_fe;

        let reactions = Self::create_reactions(json_manager, &hashmap_charge)?;
        let df_dross_fe = Self::create_dross_fe(json_manager, &hashmap_charge)?;
        let df_metal = Self::create_metal(json_manager, &reactions, &hashmap_charge, &df_dross_fe)?;
        let (df_fumes, hm_post_combustion_fumes) =
            Self::create_fumes(json_manager, &reactions, &hashmap_charge)?;
        let hm_df_in_combustion =
            df_combustion_income.generate_hashmap_str_number("components", "total")?;
        let (hm_third_dump, hm_second_dump, df_dump) = Self::create_dump(
            json_manager,
            &reactions,
            &hashmap_charge,
            &df_fumes,
            &hm_df_in_combustion,
        )?;

        let (hm_gases, hm_gases_post, df_gases_chimney) = Self::create_gases(
            json_manager,
            &reactions,
            &hashmap_charge,
            &df_fumes,
            &hm_df_in_combustion,
        )?;
        json_manager.free_c_fus = hashmap_charge["c"]
            - (reactions[0]["c"] + reactions[6]["c"] + reactions[9]["c"] + reactions[10]["c"]);

        let hm_balance_elements = Self::create_balance_elements(
            json_manager,
            &hashmap_charge,
            &hm_df_in_combustion,
            &air_hashmap,
        )?;

        Ok(FusionMatter {
            df_income_charge,
            df_combustion_income,
            reactions,
            df_dross_fe,
            df_metal,
            df_fumes,
            hm_post_combustion_fumes,
            df_dump,
            hm_second_dump,
            hm_third_dump,
            hm_gases,
            hm_gases_post,
            df_gases_chimney,
            hm_balance_elements,
        })
    }

    fn create_balance_elements(
        json_manager: &JSONManager,
        hashmap_charge: &HashMap<String, Number>,
        hashmap_df_income: &HashMap<String, Number>,
        air_hashmap: &HashMap<String, Number>,
    ) -> Result<HashMap<String, Number>, Box<dyn Error>> {
        let get_charge = |key: &str| *hashmap_charge.get(key).unwrap_or(&0.0);
        let get_df = |key: &str| *hashmap_df_income.get(key).unwrap_or(&0.0);
        let get_air = |key: &str| *air_hashmap.get(key).unwrap_or(&0.0);

        // Variables de configuración
        let init_weight = json_manager.initial_weight_dump;
        let config = &json_manager.theoretical_input.otras_configuraciones;
        let comp_initial_sn = config["composicion_escoria_inicial_sn"];
        let comp_initial_fe = config["composicion_escoria_inicial_fe"];
        let comp_initial_sio2 = config["composicion_escoria_inicial_sio2"];
        let comp_initial_al2o3 = config["composicion_escoria_inicial_al2o3"];
        let comp_initial_cao = config["composicion_escoria_inicial_cao"];
        let comp_initial_mgo = config["composicion_escoria_inicial_mgo"];
        let mgo_refractary_aport = config["aportacion_mgo_escoria"];

        // Cálculo de Sn
        let sno2 = get_charge("sno2");
        let sno = get_charge("sno");
        let sns = get_charge("sns");
        let sn = get_charge("sn");
        let fesn2 = get_charge("fesn2");
        let k25 = comp_initial_sn * init_weight / 100.0;
        let sn_value = sno2 * (118.7 / (118.7 + 16.0 * 2.0))
            + sno * (118.7 / (118.7 + 16.0))
            + sns * (118.7 / (118.7 + 32.0))
            + sn
            + fesn2 * (118.7 * 2.0 / (118.7 * 2.0 + 55.85))
            + k25;

        // Cálculo de Fe
        let sum_fe_values = json_manager.total_dist_tm_fe;
        let k26 = comp_initial_fe * init_weight / 100.0;
        let fe_value = sum_fe_values + k26;

        // Cálculo de SiO2
        let sio2 = get_charge("sio2");
        let k27 = comp_initial_sio2 * init_weight / 100.0;
        let sio2_value = sio2 + k27 + get_df("sio2");

        // Cálculo de Al2O3
        let al2o3 = get_charge("al2o3");
        let k28 = comp_initial_al2o3 * init_weight / 100.0;
        let al2o3_value = al2o3 + k28 + get_df("al2o3");

        // Cálculo de CaO
        let cao = get_charge("cao");
        let caco3 = get_charge("caco3");
        let k29 = comp_initial_cao * init_weight / 100.0;
        let cao_value = cao + caco3 * 0.56 + k29;

        // Cálculo de MgO
        let mgo = get_charge("mgo");
        let mgco3 = get_charge("mgco3");
        let k30 = comp_initial_mgo * init_weight / 100.0;
        let mgo_value = mgo
            + mgco3 * ((24.3 + 16.0) / (24.3 + 12.0 + 16.0 * 3.0))
            + k30
            + mgo_refractary_aport / 1000.0;

        // Cálculo de H2O
        let c2h6 = get_charge("c2h6");
        let h2o = get_charge("h2o");
        let h2o_value = h2o
            + get_df("h2o")
            + c2h6 * (3.0 * 18.0 / 30.0)
            + get_df("h") * (18.0 / 2.0)
            + get_air("h2o");

        // Cálculo de Otros
        let otros = get_charge("others");
        let comp_initial_otros = 100.0
            - comp_initial_sn
            - comp_initial_fe
            - comp_initial_sio2
            - comp_initial_al2o3
            - comp_initial_cao
            - comp_initial_mgo;
        let k31 = comp_initial_otros * init_weight / 100.0;
        let otros_value = otros + k31 - k25 * (16.0 / 118.7) - k26 * (16.0 / 55.85);

        // Cálculo de N
        let n_value = get_charge("n") + get_df("n") + get_air("n");

        // Cálculo de S
        let fes2 = get_charge("fes2");
        let cu2s = get_charge("cu2s");
        let s_value = get_charge("s")
            + cu2s * (32.0 / (2.0 * 63.5 + 32.0))
            + fes2 * (64.0 / (64.0 + 55.85))
            + sns * (32.0 / (118.7 + 32.0))
            + get_df("s");

        // Cálculo de C
        let c_value = caco3 * (12.0 / 100.0)
            + mgco3 * (12.0 / 84.3)
            + get_charge("c")
            + c2h6 * (24.0 / 30.0)
            + get_df("c");

        // Cálculo de O
        let fe2o3 = get_charge("fe2o3");
        let feo = get_charge("feo");
        let o_value = sno2 * (32.0 / (118.7 + 32.0))
            + sno * (16.0 / (118.7 + 16.0))
            + fe2o3 * (16.0 * 3.0 / (55.85 * 2.0 + 16.0 * 3.0))
            + feo * (16.0 / 71.85)
            + caco3 * (32.0 / 100.0)
            + mgco3 * (32.0 / 84.3)
            + k25 * (16.0 / 118.7)
            + k26 * (16.0 / 55.85)
            + get_df("o")
            + get_air("o")
            + get_charge("o");

        // Cálculo de Pb, Sb, As y Cu
        let pb_value = get_charge("pb");
        let sb_value = get_charge("sb");
        let as_value = get_charge("as");
        let cu_value = get_charge("cu") + cu2s * (2.0 * 63.5 / 159.0);

        // Se insertan los valores en el HashMap de salida
        let mut hm_balance_elements = HashMap::with_capacity(15);
        hm_balance_elements.insert("sn".to_string(), sn_value);
        hm_balance_elements.insert("fe".to_string(), fe_value);
        hm_balance_elements.insert("sio2".to_string(), sio2_value);
        hm_balance_elements.insert("al2o3".to_string(), al2o3_value);
        hm_balance_elements.insert("cao".to_string(), cao_value);
        hm_balance_elements.insert("mgo".to_string(), mgo_value);
        hm_balance_elements.insert("h2o".to_string(), h2o_value);
        hm_balance_elements.insert("otros".to_string(), otros_value);
        hm_balance_elements.insert("n".to_string(), n_value);
        hm_balance_elements.insert("s".to_string(), s_value);
        hm_balance_elements.insert("c".to_string(), c_value);
        hm_balance_elements.insert("o".to_string(), o_value);
        hm_balance_elements.insert("pb".to_string(), pb_value);
        hm_balance_elements.insert("sb".to_string(), sb_value);
        hm_balance_elements.insert("as".to_string(), as_value);
        hm_balance_elements.insert("cu".to_string(), cu_value);

        Ok(hm_balance_elements)
    }

    /// Creates the table "INGRESOS CARGA" for fusion process
    fn create_df_income_charge(json_manager: &JSONManager) -> Result<DataFrame, Box<dyn Error>> {
        let concentrate_components = [
            "sno2", "fe2o3", "fes2", "sio2", "al2o3", "cao", "pb", "sb", "as", "cu", "others",
        ];
        let mut concentrate_hashmap: HashMap<&str, Number> = HashMap::new();
        let tmh_c = json_manager.hashmap_cama["tmh"];
        let tms_c = json_manager.hashmap_cama["tms"];
        concentrate_hashmap.insert("tmh", tmh_c);
        concentrate_hashmap.insert("tms", tms_c);

        for &c_component in &concentrate_components {
            if let Some(&value) = json_manager.hashmap_cama.get(c_component) {
                concentrate_hashmap.insert(c_component, tms_c * value / 100.0);
            }
        }
        concentrate_hashmap.insert("h2o", tmh_c * json_manager.hashmap_cama["h2o"] / 100.0);

        let _df_l_w = &json_manager.df_limestone_weight;
        let _df_l_m = &json_manager.df_mineralogy_limestone;
        let limestone_components = ["fe2o3", "sio2", "al2o3", "caco3", "mgco3"];
        let mut limestones_hashmap: HashMap<&str, Number> = HashMap::new();

        let column_limestone = _df_l_w.column("tmh")?.slice(0, 6);
        let humidity_l = _df_l_w.get_conditional_value("caliza", "total", "humedad")?;
        let tmh_l = _df_l_w.get_conditional_value("caliza", "total", "tmh")?;
        let tms_l = tmh_l * (1.0 - humidity_l / 100.0);
        limestones_hashmap.insert("tmh", tmh_l);
        limestones_hashmap.insert("tms", tms_l);

        for &l_component in &limestone_components {
            let column_b = _df_l_m.column(l_component)?.slice(0, 6);
            let a_sm_prd = Self::external_sum_prd(&column_limestone, &column_b)?;
            limestones_hashmap.insert(l_component, tms_l * a_sm_prd / tmh_l / 100.0);
        }

        let column_b = _df_l_m.column("otros")?.slice(0, 6);
        let a_sm_prd = Self::external_sum_prd(&column_limestone, &column_b)?;
        limestones_hashmap.insert("others", tms_l * a_sm_prd / tmh_l / 100.0);

        limestones_hashmap.insert("h2o", tmh_l * humidity_l / 100.0);

        let recirculated = &json_manager.theoretical_input.recirculantes;
        let recirculated_list = [
            "humos",
            "d_fe_fino",
            "d_fe_zarand",
            "oxidos",
            "d_cobre",
            "d_soda",
            "escoria_h_a",
            "escoria_h_r",
            "mineral_fe",
            "conc_secundario",
            "conc_cochas",
            "arena_silice",
            "otros",
        ];
        let values: Vec<Number> = recirculated_list
            .iter()
            .map(|&i| recirculated[i][0] as Number)
            .collect();
        let _df_r_m = &json_manager.df_recirculated_mineralogy;

        let column_recirculated = Series::new("recirculated", &values);
        let recirculated_components = [
            "sno2", "sno", "sns", "sn", "fesn2", "fe2o3", "fes2", "feo", "fe", "sio2", "al2o3",
            "cao", "mgo", "pb", "sb", "as", "cu", "cu2s", "na2o", "al(oh)3",
        ];
        let mut recirculated_hashmap: HashMap<&str, Number> = HashMap::new();

        let tmh_r = json_manager.recirculated_tmh;
        let humidity_r = json_manager.recirculated_humidity;
        let tms_r = tmh_r * (1.0 - humidity_r / 100.0);
        recirculated_hashmap.insert("tmh", tmh_r);
        recirculated_hashmap.insert("tms", tms_r);

        for &r_component in &recirculated_components {
            let column_b = _df_r_m.column(r_component)?;
            let sm_prd = Self::external_sum_prd(&column_recirculated, column_b)?;
            recirculated_hashmap.insert(r_component, tms_r * sm_prd / tmh_r / 100.0);
        }

        let column_b = _df_r_m.column("otros")?;
        let sm_prd = Self::external_sum_prd(&column_recirculated, column_b)?;
        recirculated_hashmap.insert("others", tms_r * sm_prd / tmh_r / 100.0);

        recirculated_hashmap.insert("h2o", tmh_r * humidity_r / 100.0);

        let carbon_components = ["c", "o", "n", "s"];
        let mut carbon_hashmap: HashMap<&str, Number> = HashMap::new();
        let _df_carbon = &json_manager.df_carbon_feed;
        let tmh_ca = json_manager.obj_carbon_fus;

        let humidity_ca = _df_carbon.get_conditional_value("process", "fusion", "humedad")?;
        let ash_ca = _df_carbon.get_conditional_value("process", "fusion", "ceniza")?;
        let ch4_ca = _df_carbon.get_conditional_value("process", "fusion", "ch4")?;
        let tms_ca = tmh_ca * (1.0 - humidity_ca / 100.0);

        let ash_sio2 = json_manager.theoretical_input.otras_configuraciones["cenizas_sio2"];
        let ash_al2o3 = json_manager.theoretical_input.otras_configuraciones["cenizas_al2o3"];
        let ash_fe2o3 = 100.0 - (ash_sio2 + ash_al2o3);
        let fe2o3_ca = tms_ca * ash_ca / 100.0 * ash_fe2o3 / 100.0;
        let sio2_ca = tms_ca * ash_ca / 100.0 * ash_sio2 / 100.0;
        let al2o3_ca = tms_ca * ash_ca / 100.0 * ash_al2o3 / 100.0;
        let c2h6_ca = tms_ca * ch4_ca / 100.0;

        carbon_hashmap.insert("tmh", tmh_ca);
        carbon_hashmap.insert("tms", tms_ca);
        carbon_hashmap.insert("fe2o3", fe2o3_ca);
        carbon_hashmap.insert("sio2", sio2_ca);
        carbon_hashmap.insert("al2o3", al2o3_ca);
        carbon_hashmap.insert("c2h6", c2h6_ca);
        carbon_hashmap.insert("h2o", tmh_ca * humidity_ca / 100.0);

        for &ca_component in &carbon_components {
            let df_value = _df_carbon.get_conditional_value("process", "fusion", ca_component)?;
            carbon_hashmap.insert(ca_component, tms_ca * df_value / 100.0);
        }

        let molecules = [
            "tmh", "tms", "sno2", "sno", "sns", "sn", "fesn2", "fe2o3", "fes2", "feo", "fe",
            "sio2", "al2o3", "cao", "mgo", "caco3", "mgco3", "pb", "sb", "as", "cu", "cu2s",
            "na2o", "al(oh)3", "c", "c2h6", "o", "n", "s", "others", "h2o",
        ];

        let mut concentrate_vector = Vec::with_capacity(molecules.len());
        let mut limestone_vector = Vec::with_capacity(molecules.len());
        let mut recirculated_vector = Vec::with_capacity(molecules.len());
        let mut carbon_vector = Vec::with_capacity(molecules.len());

        for &mol in &molecules {
            concentrate_vector.push(*concentrate_hashmap.get(mol).unwrap_or(&0.0));
            limestone_vector.push(*limestones_hashmap.get(mol).unwrap_or(&0.0));
            recirculated_vector.push(*recirculated_hashmap.get(mol).unwrap_or(&0.0));
            carbon_vector.push(*carbon_hashmap.get(mol).unwrap_or(&0.0));
        }

        let df = DataFrame::new(vec![
            Series::new("molecules", &molecules),
            Series::new("concentrate", concentrate_vector),
            Series::new("limestone", limestone_vector),
            Series::new("recirculated", recirculated_vector),
            Series::new("carbon", carbon_vector),
        ])?;

        let lazy_df = df.lazy();
        let df_income_charge = lazy_df
            .with_column(
                (col("carbon") + col("concentrate") + col("limestone") + col("recirculated"))
                    .alias("total"),
            )
            .collect()?;

        Ok(df_income_charge)
    }

    fn create_df_combustion_income(
        json_manager: &mut JSONManager,
        df_income_charge: &DataFrame,
    ) -> Result<(DataFrame, HashMap<String, Number>), Box<dyn Error>> {
        let df_f_p = &json_manager.df_fuel_parameters;
        let df_h = &json_manager.df_heating;

        let pm = df_f_p.get_specific_value("pm", 0)?;
        let fuel = json_manager.obj_gas_fus * pm / 22.4;
        let fuel_heating = df_h.get_specific_value("combustible", 0)? * pm / 22.4;
        let time_heating = df_h.get_specific_value("tiempo", 0)?;
        let fusion_time = json_manager.fusion_time;

        let total_c = (fuel / 1000.0 * fusion_time) + (fuel_heating / 1000.0 * time_heating);

        let mut combustible_hashmap: HashMap<&str, Number> = HashMap::new();
        combustible_hashmap.insert("total", total_c);

        for c_combustible in ["c", "h", "o", "n", "s"].iter() {
            let val = df_f_p.get_specific_value(c_combustible, 0)? / 100.0 * total_c;
            combustible_hashmap.insert(c_combustible, val);
        }

        let ash_c = df_f_p.get_specific_value("ash", 0)?;
        combustible_hashmap.insert("sio2", ash_c * 2.0 / 3.0 / 100.0 * total_c);
        combustible_hashmap.insert("al2o3", ash_c * 1.0 / 3.0 / 100.0 * total_c);

        let air_humidity = json_manager.air_humidity;
        let oxygen_purity = json_manager.theoretical_input.otras_configuraciones["pureza_oxigeno"];
        let air_atomization =
            json_manager.theoretical_input.otras_configuraciones["aire_atomizacion"];
        let heating_oxygen = df_h.get_specific_value("oxigeno", 0)?;
        let heating_stoichiometry = df_h.get_specific_value("estequiom", 0)?;
        let fus_stoichiometry = json_manager.obj_fus_stoichiometry;
        let o2_fus_enrichment = json_manager.obj_o2_fus_enrichment;
        let a_air = json_manager.theoretical_input.otras_configuraciones["aire_matriz"];
        let b_oxygen = json_manager.theoretical_input.otras_configuraciones["oxigeno_matriz"];
        let c_air: Number = 0.21 * (100.0 - air_humidity) / 100.0;
        let d_oxygen: Number = oxygen_purity / 100.0;

        let fuel_factor = ((combustible_hashmap["c"] * 32.0 / 12.0)
            + (combustible_hashmap["h"] * 16.0 / 2.0)
            + combustible_hashmap["s"]
            - combustible_hashmap["o"])
            * 22.4
            / 32.0
            / c_air
            / combustible_hashmap["total"];

        json_manager.fuel_factor = fuel_factor;
        json_manager.fuel_heating = fuel_heating;
        json_manager.time_heating = time_heating;
        json_manager.heating_stoichiometry = heating_stoichiometry;
        json_manager.c_air_fus = c_air;

        let e_nm3 = (fuel_factor * fuel / 1000.0 * fusion_time * 1000.0 * fus_stoichiometry
            / 100.0
            * c_air)
            / o2_fus_enrichment
            * 100.0
            - air_atomization * fusion_time;
        let f_nm3 = fuel_factor * fuel / 1000.0 * fusion_time * 1000.0 * fus_stoichiometry / 100.0
            * c_air
            - air_atomization * fusion_time * c_air;

        let e_h_nm3 =
            (fuel_factor * fuel_heating / 1000.0 * time_heating * 1000.0 * heating_stoichiometry
                / 100.0
                * c_air)
                / heating_oxygen
                * 100.0;
        let f_h_nm3 =
            fuel_factor * fuel_heating / 1000.0 * time_heating * 1000.0 * heating_stoichiometry
                / 100.0
                * c_air;

        let denominator = a_air * d_oxygen - b_oxygen * c_air;
        let air_result_nm3 = (d_oxygen * e_nm3 - b_oxygen * f_nm3) / denominator;
        let oxygen_result_nm3 = (-c_air * e_nm3 + a_air * f_nm3) / denominator;
        let air_result_nm3_h = air_result_nm3 / fusion_time;
        let oxygen_result_nm3_h = oxygen_result_nm3 / fusion_time;

        json_manager.obj_air_result_nm3_h_fus = air_result_nm3_h;
        json_manager.obj_oxygen_result_nm3_h_fus = oxygen_result_nm3_h;

        let h_air_result_nm3 = (d_oxygen * e_h_nm3 - b_oxygen * f_h_nm3) / denominator;
        let h_oxygen_result_nm3 = (-c_air * e_h_nm3 + a_air * f_h_nm3) / denominator;

        let mut air_hashmap: HashMap<&str, Number> = HashMap::new();

        let o_a = (0.21 * (100.0 - air_humidity) / 100.0 * (air_result_nm3 + h_air_result_nm3))
            * 32.0
            / 22.4
            / 1000.0;
        let n_a = (0.21 * (100.0 - air_humidity) / 100.0 * 79.0 / 21.0
            * (air_result_nm3 + h_air_result_nm3))
            * 28.0
            / 22.4
            / 1000.0;
        let h2o_a =
            air_humidity / 100.0 * (air_result_nm3 + h_air_result_nm3) * 18.0 / 22.4 / 1000.0;

        air_hashmap.insert("o", o_a);
        air_hashmap.insert("n", n_a);
        air_hashmap.insert("h2o", h2o_a);
        air_hashmap.insert("total", o_a + n_a + h2o_a);

        let mut oxygen_hashmap: HashMap<&str, Number> = HashMap::new();
        let o_o = oxygen_purity / 100.0 * (oxygen_result_nm3 + h_oxygen_result_nm3) * 32.0
            / 22.4
            / 1000.0;
        let n_o = (1.0 - oxygen_purity / 100.0) * (oxygen_result_nm3 + h_oxygen_result_nm3) * 28.0
            / 22.4
            / 1000.0;

        oxygen_hashmap.insert("o", o_o);
        oxygen_hashmap.insert("n", n_o);
        oxygen_hashmap.insert("total", o_o + n_o);

        let mut air_atomized_hashmap: HashMap<&str, Number> = HashMap::new();
        let o_aa = 0.21 * (100.0 - air_humidity) / 100.0 * air_atomization * fusion_time * 32.0
            / 22.4
            / 1000.0;
        let n_aa = 0.21 * (100.0 - air_humidity) / 100.0 * 79.0 / 21.0
            * air_atomization
            * fusion_time
            * 28.0
            / 22.4
            / 1000.0;

        air_atomized_hashmap.insert("o", o_aa);
        air_atomized_hashmap.insert("n", n_aa);
        air_atomized_hashmap.insert("total", n_aa + o_aa);

        let components = ["total", "c", "h", "o", "n", "s", "sio2", "al2o3", "h2o"];

        let mut combustible_vector: Vec<Number> = Vec::new();
        let mut air_vector: Vec<Number> = Vec::new();
        let mut oxygen_vector: Vec<Number> = Vec::new();
        let mut aa_vector: Vec<Number> = Vec::new();

        for comp in &components {
            combustible_vector.push(*combustible_hashmap.get(comp).unwrap_or(&0.0));
            air_vector.push(*air_hashmap.get(comp).unwrap_or(&0.0));
            oxygen_vector.push(*oxygen_hashmap.get(comp).unwrap_or(&0.0));
            aa_vector.push(*air_atomized_hashmap.get(comp).unwrap_or(&0.0));
        }

        let components_series = Series::new("components", &components);
        let combustible_series = Series::new("combustible", &combustible_vector);
        let air_series = Series::new("air", &air_vector);
        let oxygen_series = Series::new("oxygen", &oxygen_vector);
        let aa_series = Series::new("a_atomized", &aa_vector);

        let df = DataFrame::new(vec![
            components_series,
            combustible_series,
            air_series,
            oxygen_series,
            aa_series,
        ])?;

        let lazy_df = df.lazy();
        let df_combustion_income = lazy_df
            .with_column(
                (col("combustible") + col("air") + col("oxygen") + col("a_atomized"))
                    .alias("total"),
            )
            .collect()?;

        let free_oxygen: Number = (fuel_factor * fuel / 1000.0
            * fusion_time
            * 1000.0
            * c_air
            * (fus_stoichiometry - 100.0)
            / 100.0)
            * 32.0
            / 22.4
            / 1000.0
            + df_income_charge.get_conditional_value("molecules", "o", "total")?;
        json_manager.free_oxygen_fus = free_oxygen;
        let air_hashmap_string: HashMap<String, Number> = air_hashmap
            .iter()
            .map(|(&k, &v)| (k.to_string(), v))
            .collect();
        Ok((df_combustion_income, air_hashmap_string))
    }
    fn create_reactions(
        json_manager: &mut JSONManager,
        hashmap_charge: &HashMap<String, Number>,
    ) -> Result<Vec<HashMap<String, Number>>, Box<dyn Error>> {
        // TODO change hardcoded atomic masses
        // TODO double check to distributions
        let mut reactions: Vec<HashMap<String, Number>> = Vec::new();
        let free_oxygen = json_manager.free_oxygen_fus;
        // reaction 1
        let ec_1_pm = hashmap_charge["sno2"];
        let ec_1 = Self::calculate_masses(ec_1_pm, &["sno2", "c"], &["sno", "co"])?;
        reactions.push(ec_1);

        // reaction 2
        let ec_2_pm = hashmap_charge["fes2"];
        let ec_2 = Self::calculate_masses(ec_2_pm, &["fes2", "sno"], &["sns", "feo", "0.5s2"])?;
        reactions.push(ec_2);

        // reaction 3
        let ec_3_pm = hashmap_charge["fesn2"];
        let ec_3 = Self::calculate_masses(ec_3_pm, &["fesn2"], &["2sn", "fe"])?;
        reactions.push(ec_3);
        // reaction 4
        let fe_ec_3 = reactions[2]["fe"];
        let ec_4_pm = if fe_ec_3 * 16. / 55.85 < free_oxygen {
            fe_ec_3 * 16. / 55.85
        } else {
            free_oxygen
        };
        let ec_4 = Self::calculate_masses(ec_4_pm, &["0.5o2", "fe"], &["feo"])?;
        reactions.push(ec_4);

        // reaction 5
        let ec_5_pm = 0.0;
        let ec_5 = Self::calculate_masses(ec_5_pm, &["0.5o2", "sn"], &["sno"])?;
        reactions.push(ec_5);

        // reaction 6
        let fe_re_6 = hashmap_charge["fe"];
        let ec_6_pm = fe_re_6 + reactions[2]["fe"] - reactions[3]["fe"];
        let ec_6 = Self::calculate_masses(ec_6_pm, &["fe", "sno"], &["feo", "sn"])?;
        reactions.push(ec_6);

        // reaction 7
        let sno_re_7 = hashmap_charge["sno"];
        let dist_gas_sno_fusion =
            json_manager.theoretical_input.otras_configuraciones["distribucion_gas_sno_fusion"];
        let distribution_sno_dump = json_manager.obj_distribution_sno_dump_fus;
        let dist_metal_dross_sno = 100. - dist_gas_sno_fusion - distribution_sno_dump;
        let ec_7_pm = (reactions[0]["sno"] + reactions[4]["sno"] + sno_re_7
            - reactions[1]["sno"]
            - reactions[5]["sno"])
            * dist_metal_dross_sno
            / 100.0;
        let ec_7 = Self::calculate_masses(ec_7_pm, &["sno", "c"], &["sn", "co"])?;
        reactions.push(ec_7);

        // reaction 8
        let ec_8_pm = (reactions[0]["sno"] + reactions[4]["sno"] + sno_re_7
            - reactions[1]["sno"]
            - reactions[5]["sno"])
            * dist_gas_sno_fusion
            / 100.;
        let ec_8 = Self::calculate_masses(ec_8_pm, &["sno"], &["sno(v)"])?;
        reactions.push(ec_8);

        // reaction 9
        let ec_9_pm = (reactions[0]["sno"] + reactions[4]["sno"] + sno_re_7
            - reactions[1]["sno"]
            - reactions[5]["sno"])
            * distribution_sno_dump
            / 100.;
        let ec_9 = Self::calculate_masses(ec_9_pm, &["sno"], &["sno(esc)"])?;
        reactions.push(ec_9);

        // reaction 10
        let ec_10_pm = hashmap_charge["fe2o3"];
        let ec_10 = Self::calculate_masses(ec_10_pm, &["fe2o3", "c"], &["2feo", "co"])?;
        reactions.push(ec_10);

        // reaction 11
        let dist_feo_fe = json_manager.obj_dist_feo_fe_fus;
        let feo_re_11 = hashmap_charge["feo"];
        let ec_11_pm = (feo_re_11
            + reactions[1]["feo"]
            + reactions[3]["feo"]
            + reactions[5]["feo"]
            + reactions[9]["2feo"])
            * dist_feo_fe
            / 100.;
        let ec_11 = Self::calculate_masses(ec_11_pm, &["feo", "c"], &["fe", "co"])?;
        reactions.push(ec_11);

        // reaction 12
        let ec_12_pm = (feo_re_11
            + reactions[1]["feo"]
            + reactions[3]["feo"]
            + reactions[5]["feo"]
            + reactions[9]["2feo"])
            * (1. - dist_feo_fe / 100.);
        let ec_12 = Self::calculate_masses(ec_12_pm, &["feo"], &["feo(esc)"])?;
        reactions.push(ec_12);

        // reaction 13
        let caco3_re_13 = hashmap_charge["caco3"];
        let ec_13 = Self::calculate_masses(caco3_re_13, &["caco3"], &["cao", "co2"])?;
        reactions.push(ec_13);

        // reaction 14
        let mgco3_re_14 = hashmap_charge["mgco3"];
        let ec_14 = Self::calculate_masses(mgco3_re_14, &["mgco3"], &["mgo", "co2"])?;
        reactions.push(ec_14);

        // reaction 15
        let cu2s_re_15 = hashmap_charge["cu2s"];
        let ec_15 = Self::calculate_masses(cu2s_re_15, &["cu2s", "sn"], &["sns", "2cu"])?;
        reactions.push(ec_15);

        // reaction 16
        let s_re_16 = hashmap_charge["s"];
        let ec_16 = Self::calculate_masses(s_re_16, &["s", "sn"], &["sns"])?;
        reactions.push(ec_16);

        // reaction 17
        let dist_o2_free_c2h6 =
            json_manager.theoretical_input.otras_configuraciones["distribucion_o2_libre_c2h6_fus"];
        let ec_17_pm = free_oxygen * dist_o2_free_c2h6 / 100.;
        let ec_17 = Self::calculate_masses(ec_17_pm, &["3.5o2", "c2h6"], &["2co2", "3h2o"])?;
        reactions.push(ec_17);

        // reaction 18
        let dist_o2_free_o2 =
            json_manager.theoretical_input.otras_configuraciones["distribucion_o2_libre_o2_fus"];
        let dist_o2_free_fe_feo = reactions[3]["0.5o2"] / free_oxygen * 100.;

        let result = if (reactions[2]["2sn"] + hashmap_charge["sn"]) * 16. / 118.7
            < (free_oxygen - reactions[3]["0.5o2"])
        {
            (reactions[2]["2sn"] + hashmap_charge["sn"]) * 16. / 118.7
        } else {
            free_oxygen - reactions[3]["0.5o2"]
        } / free_oxygen
            * 100.;

        let dump_h_r = json_manager
            .theoretical_input
            .recirculantes
            .get("escoria_h_r")
            .unwrap_or(&[0.0, 0.0])[0];
        let dist_c_co_fus = if dist_o2_free_fe_feo > 20. {
            result
        } else if dump_h_r > 30. {
            100. - dist_o2_free_fe_feo
        } else {
            0.
        };

        json_manager.obj_dist_c_co_fus = dist_c_co_fus;
        let dist_c_co = json_manager.obj_dist_c_co_fus;
        let dist_o2_free_co_co2 = 100. - (dist_o2_free_fe_feo + dist_o2_free_o2 + dist_c_co);
        let ec_18_pm = free_oxygen * dist_o2_free_co_co2 / 100.;
        let ec_18 = Self::calculate_masses(ec_18_pm, &["0.5o2", "co"], &["co2"])?;
        reactions.push(ec_18);

        // reaction 19
        let ec_19_pm = free_oxygen * dist_c_co / 100.;
        let ec_19 = Self::calculate_masses(ec_19_pm, &["0.5o2", "c"], &["co"])?;
        reactions.push(ec_19);

        // reaction 20
        let ec_20_pm = reactions[7]["sno(v)"];
        let ec_20 = Self::calculate_masses(ec_20_pm, &["sno", "0.5o2"], &["sno2"])?;
        reactions.push(ec_20);

        // reaction 21
        let ec_21_pm = hashmap_charge["sns"]
            + reactions[1]["sns"]
            + reactions[14]["sns"]
            + reactions[15]["sns"];
        let ec_21 = Self::calculate_masses(ec_21_pm, &["sns", "2o2"], &["sno2", "so2"])?;
        reactions.push(ec_21);

        // temp value 22
        reactions.push(HashMap::new());

        // reaction 23
        let ec_23_pm = reactions[1]["0.5s2"];
        let ec_23 = Self::calculate_masses(ec_23_pm, &["0.5s2", "o2"], &["so2"])?;
        reactions.push(ec_23);

        // reaction 24
        let ec_24_pm = hashmap_charge["c2h6"] - reactions[16]["c2h6"];
        let ec_24 = Self::calculate_masses(ec_24_pm, &["c2h6", "3.5o2"], &["2co2", "3h2o"])?;
        reactions.push(ec_24);

        // reaction 22
        let o2_combustion_co_co2 = (reactions[0]["co"]
            + reactions[6]["co"]
            + reactions[9]["co"]
            + reactions[10]["co"]
            + reactions[18]["co"]
            - reactions[17]["co"])
            * 16.
            / 28.;
        let calc_22 = json_manager.air_o2_fus + free_oxygen * dist_o2_free_o2 / 100.
            - reactions[19]["0.5o2"]
            - reactions[20]["2o2"]
            - reactions[22]["o2"]
            - reactions[23]["3.5o2"];
        let ec_22_pm = if calc_22 > o2_combustion_co_co2 {
            o2_combustion_co_co2
        } else {
            calc_22
        };
        let ec_22 = Self::calculate_masses(ec_22_pm, &["0.5o2", "co"], &["co2"])?;
        reactions[21] = ec_22;

        // reaction 25
        let lance_react_fus =
            json_manager.theoretical_input.otras_configuraciones["lanza_reacciona_fusion"];
        let ec_25_pm = reactions[7]["sno(v)"] * lance_react_fus;
        let ec_25 = Self::calculate_masses(ec_25_pm, &["sno", "0.5o2"], &["sno2"])?;
        reactions.push(ec_25);

        // reactions 26
        let ec_26_pm = (hashmap_charge["sns"]
            + reactions[1]["sns"]
            + reactions[14]["sns"]
            + reactions[15]["sns"])
            * lance_react_fus;
        let ec_26 = Self::calculate_masses(ec_26_pm, &["sns", "2o2"], &["sno2", "so2"])?;
        reactions.push(ec_26);

        // reactions 27
        let ec_27_pm = ec_26_pm;
        let ec_27 = Self::calculate_masses(ec_27_pm, &["0.5o2", "co"], &["co2"])?;
        reactions.push(ec_27);

        // reactions 28
        let ec_28_pm = ec_26_pm;
        let ec_28 = Self::calculate_masses(ec_28_pm, &["0.5s2", "o2"], &["so2"])?;
        reactions.push(ec_28);

        // reactions 29
        let ec_29_pm = (hashmap_charge["c2h6"] - reactions[16]["c2h6"]) * lance_react_fus;
        let ec_29 = Self::calculate_masses(ec_29_pm, &["c2h6", "3.5o2"], &["2co2", "3h2o"])?;
        reactions.push(ec_29);

        // reactions 30
        let code_react_fus =
            json_manager.theoretical_input.otras_configuraciones["codo_reacciona_fusion"];
        let ec_30_pm = reactions[7]["sno(v)"] * code_react_fus;
        let ec_30 = Self::calculate_masses(ec_30_pm, &["sno", "0.5o2"], &["sno2"])?;
        reactions.push(ec_30);

        // reactions 31
        let ec_31_pm = (hashmap_charge["sns"]
            + reactions[1]["sns"]
            + reactions[14]["sns"]
            + reactions[15]["sns"])
            * code_react_fus;
        let ec_31 = Self::calculate_masses(ec_31_pm, &["sns", "2o2"], &["sno2", "so2"])?;
        reactions.push(ec_31);

        // reactions 32
        let ec_32_pm = reactions[21]["0.5o2"] * code_react_fus;
        let ec_32 = Self::calculate_masses(ec_32_pm, &["0.5o2", "co"], &["co2"])?;
        reactions.push(ec_32);

        // reactions 33
        let ec_33_pm = reactions[1]["0.5s2"] * code_react_fus;
        let ec_33 = Self::calculate_masses(ec_33_pm, &["0.5s2", "o2"], &["so2"])?;
        reactions.push(ec_33);

        // reactions 34
        let ec_34_pm = (hashmap_charge["c2h6"] - reactions[16]["c2h6"]) * code_react_fus;
        let ec_34 = Self::calculate_masses(ec_34_pm, &["c2h6", "3.5o2"], &["2co2", "3h2o"])?;
        reactions.push(ec_34);

        Ok(reactions)
    }

    fn create_dross_fe(
        json_manager: &JSONManager,
        hashmap_charge: &HashMap<String, Number>,
    ) -> Result<DataFrame, Box<dyn Error>> {
        // Acceso directo a los valores usados varias veces
        let total_dist_tm_fe = json_manager.total_dist_tm_fe;
        let factor_dross_fusion = json_manager.theoretical_input.factor_dross_fusion;
        let df_fusion_minor_elements = &json_manager.df_fusion_minor_elements;
        let theoretical_input = &json_manager.theoretical_input.otras_configuraciones;

        // Precalcular valores minoritarios
        let dist_minor_pb =
            df_fusion_minor_elements.get_conditional_value("element", "dross fe", "pb")?;
        let dist_minor_sb =
            df_fusion_minor_elements.get_conditional_value("element", "dross fe", "sb")?;
        let dist_minor_as =
            df_fusion_minor_elements.get_conditional_value("element", "dross fe", "as")?;
        let dist_minor_cu =
            df_fusion_minor_elements.get_conditional_value("element", "dross fe", "cu")?;
        let dist_others_dross_fe_fus = theoretical_input["distribucion_otros_dross_fe_fus"];
        let dist_total_fe_dross_fe = theoretical_input["distribucion_total_fe_dross_fe"];
        let dist_fe_fus_d_fe_metal = theoretical_input["distribucion_fe_fus_d_fe_metal"];

        // Calcular valores de dross
        let fe_dross =
            (dist_total_fe_dross_fe / 100.0) * total_dist_tm_fe * (dist_fe_fus_d_fe_metal / 100.0);

        let pb_dross = hashmap_charge["pb"] * (dist_minor_pb / 100.0);
        let sb_dross = hashmap_charge["sb"] * (dist_minor_sb / 100.0);
        let as_dross = hashmap_charge["as"] * (dist_minor_as / 100.0);
        let cu_dross = (hashmap_charge["cu"]
            + hashmap_charge["cu2s"] * 2.0 * 63.5 / (2.0 * 63.5 + 32.0))
            * (dist_minor_cu / 100.0);
        let others_dross = hashmap_charge["others"] * (dist_others_dross_fe_fus / 100.0);

        let sn_dross = factor_dross_fusion
            * (fe_dross + pb_dross + sb_dross + as_dross + cu_dross + others_dross)
            / (100. - factor_dross_fusion);

        // Creación de vectores para DataFrame
        let elements = vec!["sn", "fe", "pb", "sb", "as", "cu", "others", "total"];
        let tm = vec![
            sn_dross,
            fe_dross,
            pb_dross,
            sb_dross,
            as_dross,
            cu_dross,
            others_dross,
        ];
        let total: Number = tm.iter().sum();
        let mut tm = tm;
        tm.push(total);

        // Calcular porcentajes
        let perc: Vec<Number> = tm.iter().map(|&t| t / total * 100.0).collect();

        // Crear Series y DataFrame
        let elements_series = Series::new("elements", elements);
        let tm_series = Series::new("tm", tm);
        let per_series = Series::new("perc", perc);

        let df = DataFrame::new(vec![elements_series, tm_series, per_series])?;

        Ok(df)
    }

    fn create_metal(
        json_manager: &JSONManager,
        reactions: &[HashMap<String, Number>],
        hashmap_charge: &HashMap<String, Number>,
        df_dross_fe: &DataFrame,
    ) -> Result<DataFrame, Box<dyn Error>> {
        // Acceso directo a los valores usados varias veces
        let df_fusion_minor_elements = &json_manager.df_fusion_minor_elements;
        let hashmap_dross_fe = df_dross_fe.generate_hashmap_str_number("elements", "tm")?;
        let total_dist_tm_fe = json_manager.total_dist_tm_fe;
        let theoretical_input = &json_manager.theoretical_input.otras_configuraciones;

        // Precalcular valores minoritarios
        let dist_minor_pb =
            df_fusion_minor_elements.get_conditional_value("element", "metal", "pb")?;
        let dist_minor_sb =
            df_fusion_minor_elements.get_conditional_value("element", "metal", "sb")?;
        let dist_minor_as =
            df_fusion_minor_elements.get_conditional_value("element", "metal", "as")?;
        let dist_minor_cu =
            df_fusion_minor_elements.get_conditional_value("element", "metal", "cu")?;
        let dist_others_metal = theoretical_input["distribucion_otros_metal_fus"];
        let dist_total_fe_metal = theoretical_input["distribucion_total_fe_metal"];
        let dist_fe_fus_d_fe_metal = theoretical_input["distribucion_fe_fus_d_fe_metal"];

        // Calcular valores de metal
        let sn_metal =
            reactions[6]["sn"] + reactions[5]["sn"] + reactions[2]["2sn"] + hashmap_charge["sn"]
                - reactions[4]["sn"]
                - hashmap_dross_fe["sn"]
                - reactions[14]["sn"]
                - reactions[15]["sn"];

        let fe_metal =
            (dist_total_fe_metal / 100.0) * total_dist_tm_fe * (dist_fe_fus_d_fe_metal / 100.0);
        let pb_metal = hashmap_charge["pb"] * (dist_minor_pb / 100.0);
        let sb_metal = hashmap_charge["sb"] * (dist_minor_sb / 100.0);
        let as_metal = hashmap_charge["as"] * (dist_minor_as / 100.0);
        let cu_metal = (hashmap_charge["cu"]
            + hashmap_charge["cu2s"] * 2.0 * 63.5 / (2.0 * 63.5 + 32.0))
            * (dist_minor_cu / 100.0);
        let others_metal = hashmap_charge["others"] * (dist_others_metal / 100.0);

        // Creación de vectores para DataFrame
        let elements = vec!["sn", "fe", "pb", "sb", "as", "cu", "others", "total"];
        let tm = vec![
            sn_metal,
            fe_metal,
            pb_metal,
            sb_metal,
            as_metal,
            cu_metal,
            others_metal,
        ];
        let total: Number = tm.iter().sum();
        let mut tm = tm;
        tm.push(total);

        // Calcular porcentajes
        let perc: Vec<Number> = tm.iter().map(|&t| t / total * 100.0).collect();

        // Crear Series y DataFrame
        let elements_series = Series::new("elements", elements);
        let tm_series = Series::new("tm", tm);
        let per_series = Series::new("perc", perc);

        let df = DataFrame::new(vec![elements_series, tm_series, per_series])?;

        Ok(df)
    }

    fn create_fumes(
        json_manager: &JSONManager,
        reactions: &[HashMap<String, Number>],
        hashmap_charge: &HashMap<String, Number>,
    ) -> Result<(DataFrame, HashMap<String, Number>), Box<dyn Error>> {
        // Acceso directo a los valores usados varias veces
        let theoretical_input = &json_manager.theoretical_input.otras_configuraciones;
        let df_fusion_minor_elements = &json_manager.df_fusion_minor_elements;
        let total_dist_tm_fe = json_manager.total_dist_tm_fe;

        let dist_fe_fus_fumes = theoretical_input["distribucion_fe_fus_humos"];
        let dist_total_fe_fumes = theoretical_input["distribucion_total_fe_humos"];
        let dist_others_fumes = 100.0
            - (theoretical_input["distribucion_otros_dross_fe_fus"]
                + theoretical_input["distribucion_otros_metal_fus"]
                + theoretical_input["distribucion_otros_escoria_fus"]);

        // Precalcular valores minoritarios
        let dist_minor_pb =
            df_fusion_minor_elements.get_conditional_value("element", "humos", "pb")?;
        let dist_minor_sb =
            df_fusion_minor_elements.get_conditional_value("element", "humos", "sb")?;
        let dist_minor_as =
            df_fusion_minor_elements.get_conditional_value("element", "humos", "as")?;
        let dist_minor_cu =
            df_fusion_minor_elements.get_conditional_value("element", "humos", "cu")?;

        // Calcular valores de humos
        let sno_fumes = reactions[7]["sno(v)"] - reactions[24]["sno"];
        let sno2_fumes = reactions[24]["sno2"] + reactions[25]["sno2"];
        let sns_fumes = hashmap_charge["sns"]
            + reactions[1]["sns"]
            + reactions[14]["sns"]
            + reactions[15]["sns"]
            - reactions[25]["sns"];
        let feo_fumes = total_dist_tm_fe * dist_total_fe_fumes / 100.0 * dist_fe_fus_fumes / 100.0
            * (55.85 + 16.0)
            / 55.85;
        let pb_fumes = hashmap_charge["pb"] * dist_minor_pb / 100.0;
        let sb_fumes = hashmap_charge["sb"] * dist_minor_sb / 100.0;
        let as_fumes = hashmap_charge["as"] * dist_minor_as / 100.0;
        let cu_fumes = (hashmap_charge["cu"]
            + hashmap_charge["cu2s"] * 2.0 * 63.5 / (2.0 * 63.5 + 32.0))
            * dist_minor_cu
            / 100.0;
        let other_fumes = hashmap_charge["others"] * dist_others_fumes / 100.0;

        // Creación de vectores para DataFrame
        let elements = vec![
            "sno", "sno2", "sns", "feo", "pb", "sb", "as", "cu", "others", "total",
        ];
        let mut tm = vec![
            sno_fumes,
            sno2_fumes,
            sns_fumes,
            feo_fumes,
            pb_fumes,
            sb_fumes,
            as_fumes,
            cu_fumes,
            other_fumes,
        ];
        let total: Number = tm.iter().sum();
        tm.push(total);

        // Calcular porcentajes
        let perc: Vec<Number> = tm.iter().map(|&t| t / total * 100.0).collect();

        // Crear Series y DataFrame
        let elements_series = Series::new("elements", elements);
        let tm_series = Series::new("tm", tm);
        let per_series = Series::new("perc", perc);

        let mut hp_post_combustion_fumes = HashMap::new();
        hp_post_combustion_fumes.insert(
            "sno2".to_string(),
            sno2_fumes + reactions[29]["sno2"] + reactions[30]["sno2"],
        );
        hp_post_combustion_fumes.insert("feo".to_string(), feo_fumes);
        hp_post_combustion_fumes.insert("pb".to_string(), pb_fumes);
        hp_post_combustion_fumes.insert("sb".to_string(), sb_fumes);
        hp_post_combustion_fumes.insert("as".to_string(), as_fumes);
        hp_post_combustion_fumes.insert("cu".to_string(), cu_fumes);
        hp_post_combustion_fumes.insert("others".to_string(), other_fumes);

        let df = DataFrame::new(vec![elements_series, tm_series, per_series])?;
        Ok((df, hp_post_combustion_fumes))
    }

    fn create_dump(
        json_manager: &JSONManager,
        reactions: &[HashMap<String, Number>],
        hashmap_charge: &HashMap<String, Number>,
        df_fumes: &DataFrame,
        hm_df_in_combustion: &HashMap<String, Number>,
    ) -> Result<HmHmDf, Box<dyn Error>> {
        // Acceso directo a los valores usados varias veces
        let theoretical_input = &json_manager.theoretical_input.otras_configuraciones;
        let initial_weight_dump = json_manager.initial_weight_dump;
        let df_fusion_minor_elements = &json_manager.df_fusion_minor_elements;

        // Configuraciones iniciales de composición
        let comp_initial_sn = theoretical_input["composicion_escoria_inicial_sn"];
        let comp_initial_fe = theoretical_input["composicion_escoria_inicial_fe"];
        let comp_initial_sio2 = theoretical_input["composicion_escoria_inicial_sio2"];
        let comp_initial_al2o3 = theoretical_input["composicion_escoria_inicial_al2o3"];
        let comp_initial_cao = theoretical_input["composicion_escoria_inicial_cao"];
        let comp_initial_mgo = theoretical_input["composicion_escoria_inicial_mgo"];
        let comp_initial_others = 100.0
            - (comp_initial_sn
                + comp_initial_fe
                + comp_initial_sio2
                + comp_initial_al2o3
                + comp_initial_cao
                + comp_initial_mgo);

        // Precalcular valores minoritarios
        let dist_minor_pb =
            df_fusion_minor_elements.get_conditional_value("element", "escoria", "pb")?;
        let dist_minor_sb =
            df_fusion_minor_elements.get_conditional_value("element", "escoria", "sb")?;
        let dist_minor_as =
            df_fusion_minor_elements.get_conditional_value("element", "escoria", "as")?;
        let dist_minor_cu =
            df_fusion_minor_elements.get_conditional_value("element", "escoria", "cu")?;

        // Dump 3
        let out_pb_dump = hashmap_charge["pb"] * dist_minor_pb / 100.0;
        let out_sb_dump = hashmap_charge["sb"] * dist_minor_sb / 100.0;
        let out_as_dump = hashmap_charge["as"] * dist_minor_as / 100.0;
        let out_cu_dump = (hashmap_charge["cu"]
            + hashmap_charge["cu2s"] * 2.0 * 63.5 / (2.0 * 63.5 + 32.0))
            * dist_minor_cu
            / 100.0;

        let sum_out = out_as_dump + out_cu_dump + out_sb_dump + out_pb_dump;

        let mut hm_third_dump = HashMap::new();
        hm_third_dump.insert("pb".to_string(), out_pb_dump);
        hm_third_dump.insert("sb".to_string(), out_sb_dump);
        hm_third_dump.insert("as".to_string(), out_as_dump);
        hm_third_dump.insert("cu".to_string(), out_cu_dump);
        hm_third_dump.insert("others".to_string(), sum_out);

        // Dump 1
        let dist_others_dump = theoretical_input["distribucion_otros_escoria_fus"];
        let add_mgo_dump = theoretical_input["aportacion_mgo_escoria"];

        let sno_dump_1 = reactions[8]["sno(esc)"]
            + comp_initial_sn * initial_weight_dump / 100.0 * (118.7 + 16.0) / 118.7;
        let feo_dump_1 = reactions[11]["feo(esc)"]
            - df_fumes.get_conditional_value("elements", "feo", "tm")?
            + comp_initial_fe * initial_weight_dump / 100.0 * 71.85 / 55.85;
        let sio2_dump_1 = hashmap_charge["sio2"]
            + comp_initial_sio2 * initial_weight_dump / 100.0
            + hm_df_in_combustion["sio2"];
        let al2o3_dump_1 = hashmap_charge["al2o3"]
            + comp_initial_al2o3 * initial_weight_dump / 100.0
            + hm_df_in_combustion["al2o3"];
        let cao_dump_1 = hashmap_charge["cao"]
            + reactions[12]["cao"]
            + comp_initial_cao * initial_weight_dump / 100.0;
        let mgo_dump_1 = hashmap_charge["mgo"]
            + reactions[13]["mgo"]
            + comp_initial_mgo * initial_weight_dump / 100.0
            + add_mgo_dump / 1000.0;
        let others_1 = hashmap_charge["others"] * dist_others_dump / 100.0
            + (comp_initial_others * initial_weight_dump / 100.0
                - comp_initial_sn * initial_weight_dump / 100.0 * 16.0 / 118.7
                - comp_initial_fe * initial_weight_dump / 100.0 * 16.0 / 55.85)
            + sum_out;

        let total_dump_1 = sno_dump_1
            + feo_dump_1
            + sio2_dump_1
            + al2o3_dump_1
            + cao_dump_1
            + mgo_dump_1
            + others_1;

        // Dump 2
        let sn_dump_2 = sno_dump_1 / total_dump_1 * 100.0 * 118.7 / (118.7 + 16.0);
        let feo_dump_2 = feo_dump_1 / total_dump_1 * 100.0 * 55.85 / (55.85 + 16.0);
        let others_dump_2 = others_1 / total_dump_1 * 100.0
            + sno_dump_1 / total_dump_1 * 100.0 * 16.0 / (118.7 + 16.0)
            + feo_dump_1 / total_dump_1 * 100.0 * 16.0 / (55.85 + 16.0);

        let mut hm_second_dump = HashMap::new();
        hm_second_dump.insert("sn".to_string(), sn_dump_2);
        hm_second_dump.insert("fe".to_string(), feo_dump_2);
        hm_second_dump.insert("sio2".to_string(), sio2_dump_1 / total_dump_1 * 100.0);
        hm_second_dump.insert("al2o3".to_string(), al2o3_dump_1 / total_dump_1 * 100.0);
        hm_second_dump.insert("cao".to_string(), cao_dump_1 / total_dump_1 * 100.0);
        hm_second_dump.insert("mgo".to_string(), mgo_dump_1 / total_dump_1 * 100.0);
        hm_second_dump.insert("others".to_string(), others_dump_2);

        // Crear DataFrame
        let elements = vec![
            "sno", "feo", "sio2", "al2o3", "cao", "mgo", "others", "total",
        ];
        let mut tm = vec![
            sno_dump_1,
            feo_dump_1,
            sio2_dump_1,
            al2o3_dump_1,
            cao_dump_1,
            mgo_dump_1,
            others_1,
        ];
        tm.push(total_dump_1);

        let perc: Vec<Number> = tm.iter().map(|&t| t / total_dump_1 * 100.0).collect();

        let elements_series = Series::new("elements", elements);
        let tm_series = Series::new("tm", tm);
        let per_series = Series::new("perc", perc);

        let df = DataFrame::new(vec![elements_series, tm_series, per_series])?;

        Ok((hm_third_dump, hm_second_dump, df))
    }

    fn create_gases(
        json_manager: &mut JSONManager,
        reactions: &[HashMap<String, Number>],
        hashmap_charge: &HashMap<String, Number>,
        _df_fumes: &DataFrame,
        hm_df_in_combustion: &HashMap<String, Number>,
    ) -> Result<HmHmDf, Box<dyn Error>> {
        // Acceso directo a los valores usados varias veces
        let theoretical_input = &json_manager.theoretical_input.otras_configuraciones;
        let free_oxygen = json_manager.free_oxygen_fus;
        let air_o2_inf = json_manager.air_o2_fus;
        let air_n2_inf = json_manager.air_n2_fus;
        let air_h2o_inf = json_manager.air_h2o_fus;
        let o2_others = json_manager.air_o2_others_fus;
        let n2_others = json_manager.air_n2_others_fus;
        let h2o_others = json_manager.air_h2o_others_fus;
        let fusion_time = json_manager.fusion_time;

        let lance_income_fus = theoretical_input["lanza_ingreso_fusion"];
        let lance_reaction_fus = theoretical_input["lanza_reacciona_fusion"];
        let code_income_fus = theoretical_input["codo_ingreso_fusion"];
        let code_reaction_fus = theoretical_input["codo_reacciona_fusion"];
        let dist_o2_free_o2 = theoretical_input["distribucion_o2_libre_o2_fus"];
        let fuel_factor = json_manager.fuel_factor;
        let fuel_heating = json_manager.fuel_heating;
        let time_heating = json_manager.time_heating;
        let heating_stoichiometry = json_manager.heating_stoichiometry;
        let c_air = json_manager.c_air_fus;

        // TM
        let free_oxygen_heating = fuel_factor * fuel_heating / 1000.0
            * time_heating
            * 1000.0
            * c_air
            * (heating_stoichiometry - 100.0)
            / 100.0
            * 32.0
            / 22.4
            / 1000.0;

        let co2_gases_1 = hm_df_in_combustion["c"] * 44.0 / 12.0
            + reactions[12]["co2"]
            + reactions[13]["co2"]
            + reactions[17]["co2"]
            + reactions[16]["2co2"]
            + reactions[26]["co2"]
            + reactions[28]["2co2"];

        let co_gases_1 = reactions[0]["co"]
            + reactions[6]["co"]
            + reactions[9]["co"]
            + reactions[10]["co"]
            + reactions[18]["co"]
            - reactions[17]["co"]
            - reactions[26]["co"];

        let n2_gases_1 =
            hm_df_in_combustion["n"] + hashmap_charge["n"] + air_n2_inf * lance_income_fus;

        let so2_gases_1 =
            hm_df_in_combustion["s"] * 2.0 + reactions[25]["so2"] + reactions[27]["so2"];

        let h2o_gases_1 = hm_df_in_combustion["h2o"]
            + hashmap_charge["h2o"]
            + hm_df_in_combustion["h"] * 18.0 / 2.0
            + reactions[16]["3h2o"]
            + reactions[28]["3h2o"]
            + air_h2o_inf * lance_reaction_fus;

        let c2h6_gases_1 = reactions[23]["c2h6"] - reactions[28]["c2h6"];

        let o2_gases_1 = free_oxygen * dist_o2_free_o2 / 100.0 + air_o2_inf * lance_income_fus
            - (reactions[24]["0.5o2"]
                + reactions[25]["2o2"]
                + reactions[26]["0.5o2"]
                + reactions[27]["o2"]
                + reactions[28]["3.5o2"])
            + free_oxygen_heating;

        let s2_gases_1 = reactions[1]["0.5s2"] - reactions[27]["0.5s2"];

        let elements_1 = ["co2", "co", "n2", "so2", "h2o", "c2h6", "o2", "s2", "total"];

        let mut tm_1 = vec![
            co2_gases_1,
            co_gases_1,
            n2_gases_1,
            so2_gases_1,
            h2o_gases_1,
            c2h6_gases_1,
            o2_gases_1,
            s2_gases_1,
        ];

        let total_gases_1 = tm_1.iter().sum::<Number>();
        tm_1.push(total_gases_1);

        let hm_gases: HashMap<String, Number> = elements_1
            .iter()
            .cloned()
            .map(String::from)
            .zip(tm_1.iter().cloned())
            .collect();

        // Gases post comb
        let co2_gases_2 = co2_gases_1 + reactions[31]["co2"] + reactions[33]["2co2"];
        let co_gases_2 = co_gases_1 - reactions[31]["co"];
        let n2_gases_2 = n2_gases_1 + air_n2_inf * code_income_fus;
        let so2_gases_2 = so2_gases_1 + reactions[32]["so2"] + reactions[30]["so2"];
        let h2o_gases_2 = h2o_gases_1 + reactions[33]["3h2o"] + code_reaction_fus * air_h2o_inf;
        let o2_gases_2 = air_o2_inf + free_oxygen * dist_o2_free_o2 / 100.0
            - (reactions[19]["0.5o2"]
                + reactions[20]["2o2"]
                + reactions[21]["0.5o2"]
                + reactions[22]["o2"]
                + reactions[23]["3.5o2"])
            + free_oxygen_heating;

        let elements_2 = vec!["co2", "co", "n2", "so2", "h2o", "o2", "total"];

        let mut tm_2 = vec![
            co2_gases_2,
            co_gases_2,
            n2_gases_2,
            so2_gases_2,
            h2o_gases_2,
            o2_gases_2,
        ];

        let total_gases_2 = tm_2.iter().sum::<Number>();
        tm_2.push(total_gases_2);

        let hm_gases_post: HashMap<String, Number> = elements_2
            .iter()
            .cloned()
            .map(String::from)
            .zip(tm_2.iter().cloned())
            .collect();

        // Chimney
        let o2_o5_value = co_gases_2 * 16.0 / 28.0;
        let o2_o5 = o2_o5_value.min(o2_others);
        let co_o5 = o2_o5 * 1.75;
        let co2_o5 = o2_o5 * 2.75;
        json_manager.co_o5_fus = co_o5;

        let co2_tm = co2_gases_2 + co2_o5;
        let co_tm = co_gases_2 - co_o5;
        let n2_tm = n2_gases_2 + n2_others;
        let so2_tm = so2_gases_2;
        let h2o_tm = h2o_gases_2
            + json_manager.obj_water_consumption_fus / 1000.0 * fusion_time
            + h2o_others;
        let o2_tm = o2_gases_2 + o2_others - o2_o5;

        let mut tm_3 = vec![co2_tm, co_tm, n2_tm, so2_tm, h2o_tm, o2_tm];
        let total = tm_3.iter().sum::<Number>();
        tm_3.push(total);

        let co2_nm3_h = co2_tm * 1000.0 * 22.4 / 44.0 / fusion_time;
        let co_nm3_h = co_tm * 1000.0 * 22.4 / 28.0 / fusion_time;
        let n2_nm3_h = n2_tm * 1000.0 * 22.4 / 28.0 / fusion_time;
        let so2_nm3_h = so2_tm * 1000.0 * 22.4 / 64.0 / fusion_time;
        let h2o_nm3_h = h2o_tm * 1000.0 * 22.4 / 18.0 / fusion_time;
        let o2_nm3_h = o2_tm * 1000.0 * 22.4 / 32.0 / fusion_time;

        let mut nm3_h = vec![
            co2_nm3_h, co_nm3_h, n2_nm3_h, so2_nm3_h, h2o_nm3_h, o2_nm3_h,
        ];
        let total_nm3_h = nm3_h.iter().sum::<Number>();
        nm3_h.push(total_nm3_h);

        let elements_series = Series::new("elements", elements_2);
        let tm_series = Series::new("tm", tm_3);
        let nm3_h_series = Series::new("nm3_h", nm3_h);

        let df = DataFrame::new(vec![elements_series, tm_series, nm3_h_series])?;

        Ok((hm_gases, hm_gases_post, df))
    }
}

impl Debug for FusionMatter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("FusionMatter");
        debug_struct
            .field("\ndf_income_charge", &self.df_income_charge)
            .field("\ndf_combustion_income", &self.df_combustion_income)
            .field("\ndf_dross_fe", &self.df_dross_fe)
            .field("\ndf_metal", &self.df_metal)
            .field("\ndf_fumes", &self.df_fumes)
            .field("\nhm_post_combustion_fumes", &self.hm_post_combustion_fumes)
            .field("\ndf_dump", &self.df_dump)
            .field("\nhm_second_dump", &self.hm_second_dump)
            .field("\nhm_third_dump", &self.hm_third_dump)
            .field("\nhm_gases", &self.hm_gases)
            .field("\nhm_gases_post", &self.hm_gases_post)
            .field("\ndf_gases_chimney", &self.df_gases_chimney)
            .field("\nhm_balance_elements", &self.hm_balance_elements);

        for (index, reaction) in self.reactions.iter().enumerate() {
            debug_struct.field(&format!("\nreaction[{}]", index + 1), reaction);
        }

        debug_struct.finish()
    }
}
