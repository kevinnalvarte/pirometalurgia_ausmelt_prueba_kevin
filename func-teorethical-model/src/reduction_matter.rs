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

use crate::fusion_matter::FusionMatter;
use crate::helpers::{DataFrameCustomExtensions, OptimizationHelpers};
use crate::json_manager::JSONManager;
use crate::{HmHmDf, Number};
use polars::frame::DataFrame;
use polars::prelude::{col, IntoLazy, NamedFrom, Series};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Formatter};

pub struct ReductionMatter {
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
}

impl OptimizationHelpers for ReductionMatter {}
impl ReductionMatter {
    pub fn new(
        json_manager: &mut JSONManager,
        fusion_matter: &FusionMatter,
    ) -> Result<Self, Box<dyn Error>> {
        // Self::debug_message("REDUCTION CREATION START".to_string());
        let df_income_charge = Self::create_df_income_charge(json_manager, fusion_matter)?;
        let df_combustion_income =
            Self::create_df_combustion_income(json_manager, &df_income_charge)?;

        let hashmap_charge = df_income_charge.generate_hashmap_str_number("molecules", "total")?;
        let reactions = Self::create_reactions(json_manager, &hashmap_charge)?;
        let df_dross_fe = Self::create_dross_fe(json_manager, &hashmap_charge)?;
        let df_metal = Self::create_metal(json_manager, &reactions, &hashmap_charge, &df_dross_fe)?;
        let (df_fumes, hm_post_combustion_fumes) =
            Self::create_fumes(json_manager, &reactions, &hashmap_charge)?;
        let hashmap_df_income =
            df_combustion_income.generate_hashmap_str_number("components", "total")?;
        let (hm_third_dump, hm_second_dump, df_dump) = Self::create_dump(
            json_manager,
            &reactions,
            &hashmap_charge,
            &df_fumes,
            &hashmap_df_income,
        )?;
        let (hm_gases, hm_gases_post, df_gases_chimney) = Self::create_gases(
            json_manager,
            &reactions,
            &hashmap_charge,
            &df_fumes,
            &hashmap_df_income,
        )?;

        json_manager.free_c_red = hashmap_charge["c"]
            - (reactions[0]["c"] + reactions[4]["c"] + reactions[7]["c"] + reactions[8]["c"]);

        // debug!(
        //     "**************** Ingreso Carga: **************** {:?}",
        //     df_income_charge
        // );
        // debug!(
        //     "**************** Ingreso Combustion: **************** {:?}",
        //     df_combustion_income
        // );
        // debug!(
        //     "**************** Dross Fe: **************** {:?}",
        //     df_dross_fe
        // );
        // debug!("**************** Metal: **************** {:?}", df_metal);
        // debug!("**************** Humos: **************** {:?}", df_fumes);
        // debug!(
        //     "**************** Humos post combution: **************** {:?}",
        //     hm_post_combustion_fumes
        // );
        // debug!("**************** Escoria: **************** {:?}", df_dump);
        // debug!(
        //     "**************** Escoria Secundario: **************** {:?}",
        //     hm_second_dump
        // );
        // debug!(
        //     "**************** Escoria Terciario: **************** {:?}",
        //     hm_third_dump
        // );
        // debug!("**************** Gases: **************** {:?}", hm_gases);
        // debug!(
        //     "**************** Gases Post Combustion: **************** {:?}",
        //     hm_gases_post
        // );
        // for (i, reaction) in reactions.iter().enumerate() {
        //     debug!("{:?} => {:?}", i + 1, reaction);
        // }
        // debug!(
        //     "**************** Gases Chimenea: **************** {:?}",
        //     df_gases_chimney
        // );

        // Self::debug_message("REDUCTION CREATION END".to_string());
        Ok(ReductionMatter {
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
        })
    }

    fn create_df_income_charge(
        json_manager: &mut JSONManager,
        fusion_matter: &FusionMatter,
    ) -> Result<DataFrame, Box<dyn Error>> {
        let metal_components = vec!["sn", "fe", "pb", "sb", "as", "cu", "others"];
        let df_metal_fusion = &fusion_matter.df_metal;
        let df_dump_fusion = &fusion_matter.df_dump;
        let hm_metal_fusion = df_metal_fusion.generate_hashmap_str_number("elements", "perc")?;
        let drained_metal_fusion =
            df_metal_fusion.get_conditional_value("elements", "total", "tm")?;
        let metal_drained = Number::ceil((drained_metal_fusion - 1.65) / 1.5);
        json_manager.metal_drained = metal_drained;

        // Metal
        let tmh_metal = drained_metal_fusion - metal_drained * 1.5;
        let tms_metal = tmh_metal;
        let mut metal_hashmap = HashMap::new();
        metal_hashmap.insert("tmh", tmh_metal);
        metal_hashmap.insert("tms", tms_metal);
        for element in metal_components {
            metal_hashmap.insert(element, tms_metal * hm_metal_fusion[element] / 100.);
        }

        // Dump
        let hm_dump_fusion = df_dump_fusion.generate_hashmap_str_number("elements", "perc")?;
        let hm_third_dump_fusion = &fusion_matter.hm_third_dump;
        let dump_elements = vec![
            "sno", "feo", "sio2", "al2o3", "cao", "mgo", "pb", "sb", "as", "cu",
        ];
        let tmh_dump = df_dump_fusion.get_conditional_value("elements", "total", "tm")?;
        let tms_dump = tmh_dump;
        let mut dump_hashmap = HashMap::new();
        dump_hashmap.insert("tmh", tmh_dump);
        dump_hashmap.insert("tms", tms_dump);
        for element in dump_elements {
            if hm_dump_fusion.contains_key(element) {
                dump_hashmap.insert(element, tms_dump * hm_dump_fusion[element] / 100.);
            } else {
                dump_hashmap.insert(element, hm_third_dump_fusion[element]);
            }
        }
        dump_hashmap.insert(
            "others",
            tms_dump * hm_dump_fusion["others"] / 100. - hm_third_dump_fusion["others"],
        );

        // limestones
        let _df_l_w = &json_manager.df_limestone_weight;
        let _df_l_m = &json_manager.df_mineralogy_limestone;
        let tmh_limestone: Number = 0.;
        let tms_limestone: Number = 0.;
        let limestone_tmh = _df_l_w.get_conditional_value("caliza", "total", "tmh")?;

        let limestone_components = ["fe2o3", "sio2", "al2o3", "caco3", "mgco3"];
        let mut limestones_hashmap: HashMap<&str, Number> = HashMap::new();
        let column_limestone = _df_l_w.column("tmh")?.slice(0, 6);
        limestones_hashmap.insert("tmh", tmh_limestone);
        limestones_hashmap.insert("tms", tms_limestone);
        for l_component in limestone_components {
            let column_b = _df_l_m.column(l_component)?.slice(0, 6);
            let a_sm_prd = Self::external_sum_prd(&column_limestone, &column_b)?;
            limestones_hashmap.insert(
                l_component,
                tms_limestone * a_sm_prd / limestone_tmh / 100.0,
            );
        }
        {
            let column_b = _df_l_m.column("otros")?.slice(0, 6);
            let a_sm_prd = Self::external_sum_prd(&column_limestone, &column_b)?;
            limestones_hashmap.insert("others", tms_limestone * a_sm_prd / limestone_tmh / 100.0);
        }
        let humidity_l = _df_l_w.get_conditional_value("caliza", "total", "humedad")?;
        let h2o_l = tmh_limestone * humidity_l / 100.0;
        limestones_hashmap.insert("h2o", h2o_l);

        // Recirculated
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
        let mut values: Vec<Number> = Vec::new();
        for i in recirculated_list {
            values.push(recirculated[i][0] as Number);
        }
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
        for r_component in recirculated_components {
            let column_b = _df_r_m.column(r_component)?;
            let sm_prd = Self::external_sum_prd(&column_recirculated, column_b).unwrap();
            recirculated_hashmap.insert(r_component, tms_r * sm_prd / tmh_r / 100.0);
        }
        {
            let column_b = _df_r_m.column("otros")?;
            let sm_prd = Self::external_sum_prd(&column_recirculated, column_b).unwrap();
            recirculated_hashmap.insert("others", tms_r * sm_prd / tmh_r / 100.0);
        }

        let h2o_r = tmh_r * humidity_r / 100.0;
        recirculated_hashmap.insert("h2o", h2o_r);

        // Recirculated
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
        let mut values: Vec<Number> = Vec::new();
        for i in recirculated_list {
            values.push(recirculated[i][0] as Number);
        }
        let _df_r_m = &json_manager.df_recirculated_mineralogy;

        let column_recirculated = Series::new("recirculated", &values);

        let recirculated_components = [
            "sno2", "sno", "sns", "sn", "fesn2", "fe2o3", "fes2", "feo", "fe", "sio2", "al2o3",
            "cao", "mgo", "pb", "sb", "as", "cu", "cu2s", "na2o", "al(oh)3",
        ];
        let mut recirculated_hashmap: HashMap<&str, Number> = HashMap::new();

        let tmh_r: Number = 0.;
        let recirculated_tm = json_manager.recirculated_tmh;
        let humidity_r = json_manager.recirculated_humidity;
        let tms_r: Number = 0.;
        recirculated_hashmap.insert("tmh", tmh_r);
        recirculated_hashmap.insert("tms", tms_r);
        for r_component in recirculated_components {
            let column_b = _df_r_m.column(r_component)?;
            let sm_prd = Self::external_sum_prd(&column_recirculated, column_b).unwrap();
            recirculated_hashmap.insert(r_component, tms_r * sm_prd / recirculated_tm / 100.0);
        }
        {
            let column_b = _df_r_m.column("otros")?;
            let sm_prd = Self::external_sum_prd(&column_recirculated, column_b).unwrap();
            recirculated_hashmap.insert("others", tms_r * sm_prd / recirculated_tm / 100.0);
        }

        let h2o_r = tmh_r * humidity_r / 100.0;
        recirculated_hashmap.insert("h2o", h2o_r);

        // Carbon
        let carbon_components = ["c", "o", "n", "s"];
        let mut carbon_hashmap: HashMap<&str, Number> = HashMap::new();
        let _df_carbon = &json_manager.df_carbon_feed;
        let tmh_ca = json_manager.obj_carbon_red;

        let humidity_ca = _df_carbon
            .get_conditional_value("process", "red.", "humedad")
            .unwrap();

        let ash_ca = _df_carbon.get_conditional_value("process", "red.", "ceniza")?;

        let ch4_ca = _df_carbon.get_conditional_value("process", "red.", "ch4")?;

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
        for ca_component in carbon_components {
            let df_value = _df_carbon.get_conditional_value("process", "red.", ca_component)?;
            let c_value = tms_ca * df_value / 100.0;
            carbon_hashmap.insert(ca_component, c_value);
        }

        let molecules = [
            "tmh", "tms", "sno2", "sno", "sns", "sn", "fesn2", "fe2o3", "fes2", "feo", "fe",
            "sio2", "al2o3", "cao", "mgo", "caco3", "mgco3", "pb", "sb", "as", "cu", "cu2s",
            "na2o", "al(oh)3", "c", "c2h6", "o", "n", "s", "others", "h2o",
        ];

        let mut metal_vector: Vec<Number> = Vec::new();
        let mut dump_vector: Vec<Number> = Vec::new();
        let mut limestone_vector: Vec<Number> = Vec::new();
        let mut recirculated_vector: Vec<Number> = Vec::new();
        let mut carbon_vector: Vec<Number> = Vec::new();

        for mol in molecules {
            let metal_value = dump_hashmap.get(mol).unwrap_or(&0.0);
            dump_vector.push(*metal_value);

            let dump_value = metal_hashmap.get(mol).unwrap_or(&0.0);
            metal_vector.push(*dump_value);

            let limestone_value = limestones_hashmap.get(mol).unwrap_or(&0.0);
            limestone_vector.push(*limestone_value);

            let recirculated_value = recirculated_hashmap.get(mol).unwrap_or(&0.0);
            recirculated_vector.push(*recirculated_value);

            let carbon_value = carbon_hashmap.get(mol).unwrap_or(&0.0);
            carbon_vector.push(*carbon_value);
        }
        // df_feed construction
        let molecules_series = Series::new("molecules", &molecules);
        let metal_series = Series::new("metal", metal_vector);
        let dump_series = Series::new("dump", dump_vector);
        let limestone_series = Series::new("limestone", limestone_vector);
        let recirculated_series = Series::new("recirculated", recirculated_vector);
        let carbon_series = Series::new("carbon", carbon_vector);

        let df = DataFrame::new(vec![
            molecules_series,
            metal_series,
            dump_series,
            limestone_series,
            recirculated_series,
            carbon_series,
        ])?;

        let lazy_df = df.lazy();
        let df_income_charge = lazy_df
            .with_column(
                (col("carbon")
                    + col("metal")
                    + col("dump")
                    + col("limestone")
                    + col("recirculated"))
                .alias("total"),
            )
            .collect()?;

        Ok(df_income_charge)
    }

    fn create_df_combustion_income(
        json_manager: &mut JSONManager,
        df_income_charge: &DataFrame,
    ) -> Result<DataFrame, Box<dyn Error>> {
        let df_f_p = &json_manager.df_fuel_parameters;
        let pm = df_f_p.get_specific_value("pm", 0)?;
        let fuel = json_manager.obj_gas_red * pm / 22.4;
        let reduction_time = json_manager.reduction_time;
        let red_stoichiometry = json_manager.obj_red_stoichiometry;
        let o2_red_enrichment = json_manager.obj_o2_red_enrichment;
        let air_atomization =
            json_manager.theoretical_input.otras_configuraciones["aire_atomizacion"];
        let air_humidity = json_manager.air_humidity;
        let oxygen_purity = json_manager.theoretical_input.otras_configuraciones["pureza_oxigeno"];
        let fuel_factor = json_manager.fuel_factor;

        // Combustible
        let total_c = fuel / 1000.0 * reduction_time;
        let combustible_components = ["c", "h", "o", "n", "s"];
        let mut combustible_hashmap: HashMap<&str, Number> = HashMap::new();

        combustible_hashmap.insert("total", total_c);
        for &c_combustible in &combustible_components {
            let val = df_f_p.get_specific_value(c_combustible, 0)? / 100.0 * total_c;
            combustible_hashmap.insert(c_combustible, val);
        }
        let ash_c = df_f_p.get_specific_value("ash", 0)?;
        combustible_hashmap.insert("sio2", ash_c * 2. / 3. / 100. * total_c);
        combustible_hashmap.insert("al2o3", ash_c * 1. / 3. / 100. * total_c);

        // Air
        let a_air = json_manager.theoretical_input.otras_configuraciones["aire_matriz"];
        let b_oxygen = json_manager.theoretical_input.otras_configuraciones["oxigeno_matriz"];
        let c_air = json_manager.c_air_fus;
        let d_oxygen = oxygen_purity / 100.;

        let e_nm3 = (fuel_factor * total_c * 1000. * red_stoichiometry / 100. * c_air)
            / o2_red_enrichment
            * 100.
            - air_atomization * reduction_time;
        let f_nm3 = fuel_factor * total_c * 1000. * red_stoichiometry / 100. * c_air
            - air_atomization * reduction_time * c_air;

        let denominator = a_air * d_oxygen - b_oxygen * c_air;
        let air_result_nm3 = (d_oxygen * e_nm3 - b_oxygen * f_nm3) / denominator;
        let oxygen_result_nm3 = (-c_air * e_nm3 + a_air * f_nm3) / denominator;
        let air_result_nm3_h = air_result_nm3 / reduction_time;
        let oxygen_result_nm3_h = oxygen_result_nm3 / reduction_time;

        json_manager.obj_air_result_nm3_h_red = air_result_nm3_h;
        json_manager.obj_oxygen_result_nm3_h_red = oxygen_result_nm3_h;

        let mut air_hashmap: HashMap<&str, Number> = HashMap::new();

        let o_a = (0.21 * (100. - air_humidity) / 100. * air_result_nm3) * 32. / 22.4 / 1000.;
        let n_a =
            (0.21 * (100. - air_humidity) / 100. * 79. / 21. * air_result_nm3) * 28. / 22.4 / 1000.;
        let h2o_a = air_humidity / 100. * air_result_nm3 * 18. / 22.4 / 1000.;

        air_hashmap.insert("o", o_a);
        air_hashmap.insert("n", n_a);
        air_hashmap.insert("h2o", h2o_a);
        air_hashmap.insert("total", o_a + n_a + h2o_a);

        // Oxygen
        let mut oxygen_hashmap: HashMap<&str, Number> = HashMap::new();
        let o_o = oxygen_purity / 100. * oxygen_result_nm3 * 32. / 22.4 / 1000.;
        let n_o = (1. - oxygen_purity / 100.) * oxygen_result_nm3 * 28. / 22.4 / 1000.;
        oxygen_hashmap.insert("o", o_o);
        oxygen_hashmap.insert("n", n_o);
        oxygen_hashmap.insert("total", o_o + n_o);

        // Air Atomized
        let mut air_atomized_hashmap: HashMap<&str, Number> = HashMap::new();
        let o_aa = 0.21 * (100. - air_humidity) / 100. * air_atomization * reduction_time * 32.
            / 22.4
            / 1000.;
        let n_aa = 0.21 * (100. - air_humidity) / 100. * 79. / 21.
            * air_atomization
            * reduction_time
            * 28.
            / 22.4
            / 1000.;
        air_atomized_hashmap.insert("o", o_aa);
        air_atomized_hashmap.insert("n", n_aa);
        air_atomized_hashmap.insert("total", n_aa + o_aa);

        // DataFrame creation
        let components = ["total", "c", "h", "o", "n", "s", "sio2", "al2o3", "h2o"];

        let mut combustible_vector = Vec::with_capacity(components.len());
        let mut air_vector = Vec::with_capacity(components.len());
        let mut oxygen_vector = Vec::with_capacity(components.len());
        let mut aa_vector = Vec::with_capacity(components.len());

        for &comp in &components {
            combustible_vector.push(*combustible_hashmap.get(comp).unwrap_or(&0.0));
            air_vector.push(*air_hashmap.get(comp).unwrap_or(&0.0));
            oxygen_vector.push(*oxygen_hashmap.get(comp).unwrap_or(&0.0));
            aa_vector.push(*air_atomized_hashmap.get(comp).unwrap_or(&0.0));
        }

        let components_series = Series::new("components", &components);
        let combustible_series = Series::new("combustible", combustible_vector);
        let air_series = Series::new("air", air_vector);
        let oxygen_series = Series::new("oxygen", oxygen_vector);
        let aa_series = Series::new("a_atomized", aa_vector);

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

        let free_oxygen =
            (fuel_factor * total_c * 1000. * c_air * (red_stoichiometry - 100.) / 100.) * 32.
                / 22.4
                / 1000.
                + df_income_charge.get_conditional_value("molecules", "o", "total")?;
        json_manager.free_oxygen_red = free_oxygen;

        Ok(df_combustion_income)
    }

    fn create_reactions(
        json_manager: &JSONManager,
        hashmap_charge: &HashMap<String, Number>,
    ) -> Result<Vec<HashMap<String, Number>>, Box<dyn Error>> {
        let mut reactions: Vec<HashMap<String, Number>> = Vec::new();

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
        let ec_4_pm = hashmap_charge["fe"] + reactions[2]["fe"];
        let ec_4 = Self::calculate_masses(ec_4_pm, &["fe", "sno"], &["feo", "sn"])?;
        reactions.push(ec_4);

        // reaction 5
        let dist_gas_sno_red =
            json_manager.theoretical_input.otras_configuraciones["distribucion_gas_sno_reduccion"];
        let dist_sno_dump_red = json_manager.obj_distribution_sno_dump_red;
        let dist_sno_metal_red = 100. - (dist_gas_sno_red + dist_sno_dump_red);
        let mut ec_5_pm =
            hashmap_charge["sno"] + reactions[0]["sno"] - reactions[1]["sno"] - reactions[3]["sno"];
        ec_5_pm *= dist_sno_metal_red / 100.;
        let ec_5 = Self::calculate_masses(ec_5_pm, &["sno", "c"], &["sn", "co"])?;
        reactions.push(ec_5);

        // reaction 6
        let mut ec_6_pm =
            hashmap_charge["sno"] + reactions[0]["sno"] - reactions[1]["sno"] - reactions[3]["sno"];
        ec_6_pm *= dist_gas_sno_red / 100.;
        let ec_6 = Self::calculate_masses(ec_6_pm, &["sno"], &["sno(v)"])?;
        reactions.push(ec_6);

        // reactions 7
        let mut ec_7_pm =
            hashmap_charge["sno"] + reactions[0]["sno"] - reactions[1]["sno"] - reactions[3]["sno"];
        ec_7_pm *= dist_sno_dump_red / 100.;
        let ec_7 = Self::calculate_masses(ec_7_pm, &["sno"], &["sno(esc)"])?;
        reactions.push(ec_7);

        // reactions 8
        let ec_8_pm = hashmap_charge["fe2o3"];
        let ec_8 = Self::calculate_masses(ec_8_pm, &["fe2o3", "c"], &["2feo", "co"])?;
        reactions.push(ec_8);

        // reactions 9
        let dist_feo_fe_red = json_manager.obj_dist_feo_fe_red;
        let mut ec_9_pm = hashmap_charge["feo"]
            + reactions[1]["feo"]
            + reactions[3]["feo"]
            + reactions[7]["2feo"];
        ec_9_pm *= dist_feo_fe_red / 100.;
        let ec_9 = Self::calculate_masses(ec_9_pm, &["feo", "c"], &["fe", "co"])?;
        reactions.push(ec_9);

        // reaction 10
        let mut ec_10_pm = hashmap_charge["feo"]
            + reactions[1]["feo"]
            + reactions[3]["feo"]
            + reactions[7]["2feo"];
        ec_10_pm *= 1. - dist_feo_fe_red / 100.;
        let ec_10 = Self::calculate_masses(ec_10_pm, &["feo"], &["feo(esc)"])?;
        reactions.push(ec_10);

        // reaction 11
        let ec_11_pm = hashmap_charge["caco3"];
        let ec_11 = Self::calculate_masses(ec_11_pm, &["caco3"], &["cao", "co2"])?;
        reactions.push(ec_11);

        // reaction 12
        let ec_12_pm = hashmap_charge["mgco3"];
        let ec_12 = Self::calculate_masses(ec_12_pm, &["mgco3"], &["mgo", "co2"])?;
        reactions.push(ec_12);

        // reactions 13
        let ec_13_pm = hashmap_charge["cu2s"];
        let ec_13 = Self::calculate_masses(ec_13_pm, &["cu2s", "sn"], &["sns", "2cu"])?;
        reactions.push(ec_13);

        // reactions 14
        let ec_14_pm = hashmap_charge["s"];
        let ec_14 = Self::calculate_masses(ec_14_pm, &["s", "sn"], &["sns"])?;
        reactions.push(ec_14);

        // reaction 15
        // TODO: Review c2f6 values pointing to fe->feo
        // TODO: Review Negative free Oxygen
        let free_oxygen = json_manager.free_oxygen_red;
        let dist_o2_free_c2h6_red =
            json_manager.theoretical_input.otras_configuraciones["distribucion_o2_libre_c2h6_red"];
        let ec_15_pm = free_oxygen * dist_o2_free_c2h6_red / 100.;
        let ec_15 = Self::calculate_masses(ec_15_pm, &["3.5o2", "c2h6"], &["2co2", "3h2o"])?;
        reactions.push(ec_15);

        // reaction 16
        let dist_o2_free_o2_red =
            json_manager.theoretical_input.otras_configuraciones["distribucion_o2_libre_o2_red"];
        let dist_c_co_red = json_manager.obj_dist_c_co_red;
        let dist_co_co2_red = 100. - (dist_o2_free_o2_red + dist_c_co_red + dist_o2_free_c2h6_red);
        let ec_16_pm = free_oxygen * dist_co_co2_red / 100.;
        let ec_16 = Self::calculate_masses(ec_16_pm, &["0.5o2", "co"], &["co2"])?;
        reactions.push(ec_16);

        // reaction 17
        let ec_17_pm = free_oxygen * dist_c_co_red / 100.;
        let ec_17 = Self::calculate_masses(ec_17_pm, &["0.5o2", "c"], &["co"])?;
        reactions.push(ec_17);

        // reaction 18
        let ec_18_pm = reactions[5]["sno(v)"];
        let ec_18 = Self::calculate_masses(ec_18_pm, &["sno", "0.5o2"], &["sno2"])?;
        reactions.push(ec_18);

        // reaction 19
        let ec_19_pm = hashmap_charge["sns"]
            + reactions[1]["sns"]
            + reactions[12]["sns"]
            + reactions[13]["sns"];
        let ec_19 = Self::calculate_masses(ec_19_pm, &["sns", "2o2"], &["sno2", "so2"])?;
        reactions.push(ec_19);

        // temp reaction 20
        reactions.push(HashMap::new());

        // reaction 21
        let ec_21_pm = reactions[1]["0.5s2"];
        let ec_21 = Self::calculate_masses(ec_21_pm, &["0.5s2", "o2"], &["so2"])?;
        reactions.push(ec_21);

        // reaction 22
        let ec_22_pm = hashmap_charge["c2h6"] - reactions[14]["c2h6"];
        let ec_22 = Self::calculate_masses(ec_22_pm, &["c2h6", "3.5o2"], &["2co2", "3h2o"])?;
        reactions.push(ec_22);

        // reaction 20
        let calc_20 = json_manager.air_o2_red + free_oxygen * dist_o2_free_o2_red / 100.
            - reactions[17]["0.5o2"]
            - reactions[18]["2o2"]
            - reactions[20]["o2"]
            - reactions[21]["3.5o2"];
        let o2_combustion_co_co2 = (reactions[0]["co"]
            + reactions[4]["co"]
            + reactions[7]["co"]
            + reactions[8]["co"]
            + reactions[16]["co"]
            - reactions[15]["co"])
            * 16.
            / 28.;
        let ec_20_pm = if calc_20 > o2_combustion_co_co2 {
            o2_combustion_co_co2
        } else {
            calc_20
        };
        let ec_20 = Self::calculate_masses(ec_20_pm, &["0.5o2", "co"], &["co2"])?;
        reactions[19] = ec_20;

        // reaction 23
        let lance_reaction_red =
            json_manager.theoretical_input.otras_configuraciones["lanza_reacciona_reduccion"];
        let ec_23_pm = reactions[5]["sno(v)"] * lance_reaction_red;
        let ec_23 = Self::calculate_masses(ec_23_pm, &["sno", "0.5o2"], &["sno2"])?;
        reactions.push(ec_23);

        // reaction 24
        let mut ec_24_pm = hashmap_charge["sns"]
            + reactions[1]["sns"]
            + reactions[12]["sns"]
            + reactions[13]["sns"];
        ec_24_pm *= lance_reaction_red;
        let ec_24 = Self::calculate_masses(ec_24_pm, &["sns", "2o2"], &["sno2", "so2"])?;
        reactions.push(ec_24);

        // reaction 25
        let ec_25_pm = if calc_20 > o2_combustion_co_co2 {
            o2_combustion_co_co2 * lance_reaction_red
        } else {
            calc_20 * lance_reaction_red
        };
        let ec_25 = Self::calculate_masses(ec_25_pm, &["0.5o2", "co"], &["co2"])?;
        reactions.push(ec_25);

        // reaction 26
        let ec_26_pm = reactions[1]["0.5s2"] * lance_reaction_red;
        let ec_26 = Self::calculate_masses(ec_26_pm, &["0.5s2", "o2"], &["so2"])?;
        reactions.push(ec_26);

        // reaction 27
        let ec_27_pm = (hashmap_charge["c2h6"] - reactions[14]["c2h6"]) * lance_reaction_red;
        let ec_27 = Self::calculate_masses(ec_27_pm, &["c2h6", "3.5o2"], &["2co2", "3h2o"])?;
        reactions.push(ec_27);

        // reaction 28
        let code_reaction_red =
            json_manager.theoretical_input.otras_configuraciones["codo_reacciona_reduccion"];
        let ec_28_pm = reactions[5]["sno(v)"] * code_reaction_red;
        let ec_28 = Self::calculate_masses(ec_28_pm, &["sno", "0.5o2"], &["sno2"])?;
        reactions.push(ec_28);

        // reaction 29
        let ec_29_pm = hashmap_charge["sns"]
            + reactions[1]["sns"]
            + reactions[12]["sns"]
            + reactions[13]["sns"];
        let ec_29 = Self::calculate_masses(ec_29_pm, &["sns", "2o2"], &["sno2", "so2"])?;
        reactions.push(ec_29);

        // reactions 30
        let ec_30_pm = if calc_20 > o2_combustion_co_co2 {
            o2_combustion_co_co2 * code_reaction_red
        } else {
            calc_20 * code_reaction_red
        };
        let ec_30 = Self::calculate_masses(ec_30_pm, &["0.5o2", "co"], &["co2"])?;
        reactions.push(ec_30);

        // reactions 31
        let ec_31_pm = reactions[1]["0.5s2"] * code_reaction_red;
        let ec_31 = Self::calculate_masses(ec_31_pm, &["0.5s2", "o2"], &["so2"])?;
        reactions.push(ec_31);

        // reactions 32
        let ec_32_pm = (hashmap_charge["c2h6"] + reactions[14]["c2h6"]) * code_reaction_red;
        let ec_32 = Self::calculate_masses(ec_32_pm, &["c2h6", "3.5o2"], &["2co2", "3h2o"])?;
        reactions.push(ec_32);

        Ok(reactions)
    }

    fn create_dross_fe(
        json_manager: &JSONManager,
        hashmap_charge: &HashMap<String, Number>,
    ) -> Result<DataFrame, Box<dyn Error>> {
        let total_dist_tm_fe = json_manager.total_dist_tm_fe;
        let df_reduction_minor_elements = &json_manager.df_reduction_minor_elements;
        let dist_minor_pb =
            df_reduction_minor_elements.get_conditional_value("element", "dross fe", "pb")?;
        let dist_minor_sb =
            df_reduction_minor_elements.get_conditional_value("element", "dross fe", "sb")?;
        let dist_minor_as =
            df_reduction_minor_elements.get_conditional_value("element", "dross fe", "as")?;
        let dist_minor_cu =
            df_reduction_minor_elements.get_conditional_value("element", "dross fe", "cu")?;
        let dist_others_dross_fe_red =
            json_manager.theoretical_input.otras_configuraciones["distribucion_otros_dross_fe_red"];
        let dist_total_fe_dross_fe =
            json_manager.theoretical_input.otras_configuraciones["distribucion_total_fe_dross_fe"];
        let factor_dross_reduction = json_manager.theoretical_input.factor_dross_reduccion;
        let dist_fe_fus_d_fe_metal =
            json_manager.theoretical_input.otras_configuraciones["distribucion_fe_fus_d_fe_metal"];
        let dist_fe_red_d_fe_metal = 100. - dist_fe_fus_d_fe_metal;

        // tm
        let fe_dross =
            dist_total_fe_dross_fe / 100. * total_dist_tm_fe * dist_fe_red_d_fe_metal / 100.;
        let pb_dross = hashmap_charge["pb"] * dist_minor_pb / 100.;
        let sb_dross = hashmap_charge["sb"] * dist_minor_sb / 100.;
        let as_dross = hashmap_charge["as"] * dist_minor_as / 100.;
        let cu_dross = (hashmap_charge["cu"]
            + hashmap_charge["cu2s"] * 2. * 63.5 / (2. * 63.5 + 32.))
            * dist_minor_cu
            / 100.;
        let others_dross = hashmap_charge["others"] * dist_others_dross_fe_red / 100.;

        let sn_dross = factor_dross_reduction
            * (fe_dross + pb_dross + sb_dross + as_dross + cu_dross + others_dross)
            / (100. - factor_dross_reduction);

        let mut elements = vec!["sn", "fe", "pb", "sb", "as", "cu", "others"];
        let mut tm = vec![
            sn_dross,
            fe_dross,
            pb_dross,
            sb_dross,
            as_dross,
            cu_dross,
            others_dross,
        ];
        let mut perc: Vec<Number> = Vec::new();
        let total = tm.iter().sum::<Number>();
        for t in &tm {
            perc.push(t / total * 100.);
        }

        elements.push("total");
        tm.push(total);
        perc.push(100.);

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
        let df_reduction_minor_elements = &json_manager.df_reduction_minor_elements;
        let hashmap_dross_fe = df_dross_fe.generate_hashmap_str_number("elements", "tm")?;
        let total_dist_tm_fe = json_manager.total_dist_tm_fe;
        let dist_fe_fus_d_fe_metal =
            json_manager.theoretical_input.otras_configuraciones["distribucion_fe_fus_d_fe_metal"];
        let dist_fe_red_d_fe_metal = 100. - dist_fe_fus_d_fe_metal;
        let dist_others_metal =
            json_manager.theoretical_input.otras_configuraciones["distribucion_otros_metal_red"];
        let dist_total_fe_metal =
            json_manager.theoretical_input.otras_configuraciones["distribucion_total_fe_metal"];
        let dist_minor_pb =
            df_reduction_minor_elements.get_conditional_value("element", "metal", "pb")?;
        let dist_minor_sb =
            df_reduction_minor_elements.get_conditional_value("element", "metal", "sb")?;
        let dist_minor_as =
            df_reduction_minor_elements.get_conditional_value("element", "metal", "as")?;
        let dist_minor_cu =
            df_reduction_minor_elements.get_conditional_value("element", "metal", "cu")?;

        let sn_metal =
            reactions[4]["sn"] + reactions[3]["sn"] + reactions[2]["2sn"] + hashmap_charge["sn"]
                - hashmap_dross_fe["sn"]
                - reactions[12]["sn"]
                - reactions[13]["sn"];

        let fe_metal =
            dist_total_fe_metal * total_dist_tm_fe / 100. * dist_fe_red_d_fe_metal / 100.;
        let pb_metal = hashmap_charge["pb"] * dist_minor_pb / 100.;
        let sb_metal = hashmap_charge["sb"] * dist_minor_sb / 100.;
        let as_metal = hashmap_charge["as"] * dist_minor_as / 100.;
        let cu_metal = (hashmap_charge["cu"]
            + hashmap_charge["cu2s"] * 2. * 63.5 / (2. * 63.5 + 32.))
            * dist_minor_cu
            / 100.;
        let others_metal = hashmap_charge["others"] * dist_others_metal / 100.;

        let mut elements = vec!["sn", "fe", "pb", "sb", "as", "cu", "others"];
        let mut tm = vec![
            sn_metal,
            fe_metal,
            pb_metal,
            sb_metal,
            as_metal,
            cu_metal,
            others_metal,
        ];
        let mut perc: Vec<Number> = Vec::new();
        let total = tm.iter().sum::<Number>();
        for t in &tm {
            perc.push(t / total * 100.);
        }

        elements.push("total");
        tm.push(total);
        perc.push(100.);

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
        let dist_fe_fus_fumes =
            json_manager.theoretical_input.otras_configuraciones["distribucion_fe_fus_humos"];
        let dist_fe_fus_fumes = 100. - dist_fe_fus_fumes;
        let dist_total_fe_fumes =
            json_manager.theoretical_input.otras_configuraciones["distribucion_total_fe_humos"];
        let total_dist_tm_fe = json_manager.total_dist_tm_fe;
        let df_reduction_minor_elements = &json_manager.df_reduction_minor_elements;
        let dist_others_fumes = 100.
            - (json_manager.theoretical_input.otras_configuraciones
                ["distribucion_otros_dross_fe_red"]
                + json_manager.theoretical_input.otras_configuraciones
                    ["distribucion_otros_metal_red"]
                + json_manager.theoretical_input.otras_configuraciones
                    ["distribucion_otros_escoria_red"]);

        let dist_minor_pb =
            df_reduction_minor_elements.get_conditional_value("element", "humos", "pb")?;
        let dist_minor_sb =
            df_reduction_minor_elements.get_conditional_value("element", "humos", "sb")?;
        let dist_minor_as =
            df_reduction_minor_elements.get_conditional_value("element", "humos", "as")?;
        let dist_minor_cu =
            df_reduction_minor_elements.get_conditional_value("element", "humos", "cu")?;

        let sno_fumes = reactions[5]["sno(v)"] - reactions[22]["sno"];
        let sno2_fumes = reactions[22]["sno2"] + reactions[23]["sno2"];
        let sns_fumes = hashmap_charge["sns"]
            + reactions[1]["sns"]
            + reactions[12]["sns"]
            + reactions[13]["sns"]
            - reactions[23]["sns"];
        let feo_fumes = total_dist_tm_fe * dist_total_fe_fumes / 100. * dist_fe_fus_fumes / 100.
            * (55.85 + 16.)
            / 55.85;
        let pb_fumes = hashmap_charge["pb"] * dist_minor_pb / 100.;
        let sb_fumes = hashmap_charge["sb"] * dist_minor_sb / 100.;
        let as_fumes = hashmap_charge["as"] * dist_minor_as / 100.;
        let cu_fumes = (hashmap_charge["cu"]
            + hashmap_charge["cu2s"] * 2. * 63.5 / (2. * 63.5 + 32.))
            * dist_minor_cu
            / 100.;
        let other_fumes = hashmap_charge["others"] * dist_others_fumes / 100.;

        let mut elements = vec![
            "sno", "sno2", "sns", "feo", "pb", "sb", "as", "cu", "others",
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
        let mut perc: Vec<Number> = Vec::new();
        let total = tm.iter().sum::<Number>();
        for t in &tm {
            perc.push(t / total * 100.);
        }

        elements.push("total");
        tm.push(total);
        perc.push(100.);

        let elements_series = Series::new("elements", elements);
        let tm_series = Series::new("tm", tm);
        let per_series = Series::new("perc", perc);

        let mut hp_post_combustion_fumes = HashMap::new();
        hp_post_combustion_fumes.insert(
            "sno2".to_string(),
            sno2_fumes + reactions[27]["sno2"] + reactions[28]["sno2"],
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
        let df_reduction_minor_elements = &json_manager.df_reduction_minor_elements;
        let dist_minor_pb =
            df_reduction_minor_elements.get_conditional_value("element", "escoria", "pb")?;
        let dist_minor_sb =
            df_reduction_minor_elements.get_conditional_value("element", "escoria", "sb")?;
        let dist_minor_as =
            df_reduction_minor_elements.get_conditional_value("element", "escoria", "as")?;

        // Dump 3
        let out_pb_dump = hashmap_charge["pb"] * dist_minor_pb / 100.;
        let out_sb_dump = hashmap_charge["sb"] * dist_minor_sb / 100.;
        let out_as_dump = hashmap_charge["as"] * dist_minor_as / 100.;
        let out_cu_dump = 0.0;

        let mut hm_third_dump = HashMap::new();
        hm_third_dump.insert("pb".to_string(), out_pb_dump);
        hm_third_dump.insert("sb".to_string(), out_sb_dump);
        hm_third_dump.insert("as".to_string(), out_as_dump);
        hm_third_dump.insert("cu".to_string(), out_cu_dump);
        hm_third_dump.insert(
            "others".to_string(),
            out_cu_dump + out_sb_dump + out_as_dump + out_pb_dump,
        );

        // Dump 1
        let dist_others_dump =
            json_manager.theoretical_input.otras_configuraciones["distribucion_otros_escoria_red"];

        let sno_dump_1 = reactions[6]["sno(esc)"];
        let feo_dump_1 =
            reactions[9]["feo(esc)"] - df_fumes.get_conditional_value("elements", "feo", "tm")?;
        let sio2_dump_1 = hashmap_charge["sio2"] + hm_df_in_combustion["sio2"];
        let al2o3_dump_1 = hashmap_charge["al2o3"] + hm_df_in_combustion["al2o3"];
        let cao_dump_1 = hashmap_charge["cao"] + reactions[10]["cao"];
        let mgo_dump_1 = hashmap_charge["mgo"] + reactions[11]["mgo"];
        let others_1 = hashmap_charge["others"] * dist_others_dump / 100.;
        let total_dump_1 = sno_dump_1
            + feo_dump_1
            + sio2_dump_1
            + al2o3_dump_1
            + cao_dump_1
            + mgo_dump_1
            + others_1;

        // Dump 2
        let sn_dump_2 = sno_dump_1 / total_dump_1 * 100. * 118.7 / (118.7 + 16.);

        let feo_dump_2 = feo_dump_1 / total_dump_1 * 100. * 55.85 / (55.85 + 16.);

        let others_dump_2 = others_1 / total_dump_1 * 100.
            + sno_dump_1 / total_dump_1 * 100. * 16. / (118.7 + 16.0)
            + feo_dump_1 / total_dump_1 * 100. * 16. / (55.85 + 16.);

        let mut hm_second_dump = HashMap::new();
        hm_second_dump.insert("sn".to_string(), sn_dump_2);
        hm_second_dump.insert("fe".to_string(), feo_dump_2);
        hm_second_dump.insert("sio2".to_string(), sio2_dump_1 / total_dump_1 * 100.);
        hm_second_dump.insert("al2o3".to_string(), al2o3_dump_1 / total_dump_1 * 100.);
        hm_second_dump.insert("cao".to_string(), cao_dump_1 / total_dump_1 * 100.);
        hm_second_dump.insert("mgo".to_string(), mgo_dump_1 / total_dump_1 * 100.);
        hm_second_dump.insert("others".to_string(), others_dump_2);

        let mut elements = vec!["sno", "feo", "sio2", "al2o3", "cao", "mgo", "others"];
        let mut tm = vec![
            sno_dump_1,
            feo_dump_1,
            sio2_dump_1,
            al2o3_dump_1,
            cao_dump_1,
            mgo_dump_1,
            others_1,
        ];
        let total = tm.iter().sum::<Number>();
        let mut perc: Vec<Number> = Vec::new();

        for t in &tm {
            perc.push(t / total * 100.);
        }

        elements.push("total");
        tm.push(total);
        perc.push(100.);

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
        let free_oxygen = json_manager.free_oxygen_red;
        let air_o2_inf = json_manager.air_o2_red;
        let air_n2_inf = json_manager.air_n2_red;
        let air_h2o_inf = json_manager.air_h2o_red;
        let o2_others = json_manager.air_o2_others_red;
        let n2_others = json_manager.air_n2_others_red;
        let h2o_others = json_manager.air_h2o_others_red;
        let lance_income_red =
            json_manager.theoretical_input.otras_configuraciones["lanza_ingreso_reduccion"];
        let lance_reaction_red =
            json_manager.theoretical_input.otras_configuraciones["lanza_reacciona_reduccion"];
        let code_income_red =
            json_manager.theoretical_input.otras_configuraciones["codo_ingreso_reduccion"];
        let code_reaction_red =
            json_manager.theoretical_input.otras_configuraciones["codo_reacciona_reduccion"];
        let reduction_time = json_manager.reduction_time;

        let dist_o2_free_o2 =
            json_manager.theoretical_input.otras_configuraciones["distribucion_o2_libre_o2_red"];

        // TM
        let co2_gases_1 = hm_df_in_combustion["c"] * 44. / 12.
            + reactions[10]["co2"]
            + reactions[11]["co2"]
            + reactions[15]["co2"]
            + reactions[14]["2co2"]
            + reactions[24]["co2"]
            + reactions[14]["2co2"];

        let co_gases_1 = reactions[0]["co"]
            + reactions[4]["co"]
            + reactions[7]["co"]
            + reactions[8]["co"]
            + reactions[16]["co"]
            - reactions[15]["co"]
            - reactions[24]["co"];

        let n2_gases_1 =
            hm_df_in_combustion["n"] + hashmap_charge["n"] + air_n2_inf * lance_reaction_red;

        let so2_gases_1 =
            hm_df_in_combustion["s"] * 2. + reactions[23]["so2"] + reactions[25]["so2"];
        let h2o_gases_1 = hm_df_in_combustion["h2o"]
            + hashmap_charge["h2o"]
            + hm_df_in_combustion["h"] * 18. / 2.
            + reactions[14]["3h2o"]
            + reactions[26]["3h2o"]
            + air_h2o_inf * lance_reaction_red;

        let c2h6_gases_1 = reactions[21]["c2h6"] - reactions[26]["c2h6"];
        let o2_gases_1 = free_oxygen * 0. / 100. + dist_o2_free_o2 * lance_income_red
            - (reactions[22]["0.5o2"]
                + reactions[23]["2o2"]
                + reactions[24]["0.5o2"]
                + reactions[25]["o2"]
                + reactions[26]["3.5o2"]);
        let s2_gases_1 = reactions[1]["0.5s2"] - reactions[25]["0.5s2"];

        let elements_1 = vec![
            "co2".to_string(),
            "co".to_string(),
            "n2".to_string(),
            "so2".to_string(),
            "h2o".to_string(),
            "c2h6".to_string(),
            "o2".to_string(),
            "s2".to_string(),
            "total".to_string(),
        ];

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

        let hm_gases: HashMap<String, Number> = elements_1.into_iter().zip(tm_1).collect();

        // Gases 2 post comb
        let co2_gases_2 = co2_gases_1 + reactions[29]["co2"] + reactions[31]["2co2"];
        let co_gases_2 = co_gases_1 - reactions[29]["co"];
        let n2_gases_2 = n2_gases_1 + air_n2_inf * code_income_red;
        let so2_gases_2 = so2_gases_1 + reactions[30]["so2"] + reactions[28]["so2"];
        let h2o_gases_2 = h2o_gases_1 + reactions[31]["3h2o"] + air_h2o_inf * code_reaction_red;
        let o2_gases_2 = air_o2_inf + free_oxygen * dist_o2_free_o2 / 100.
            - reactions[17]["0.5o2"]
            - reactions[18]["2o2"]
            - reactions[19]["0.5o2"]
            - reactions[20]["o2"]
            - reactions[21]["3.5o2"];
        let elements_2 = vec![
            "co2".to_string(),
            "co".to_string(),
            "n2".to_string(),
            "so2".to_string(),
            "h2o".to_string(),
            "o2".to_string(),
            "total".to_string(),
        ];

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

        let hm_gases_post: HashMap<String, Number> =
            elements_2.clone().into_iter().zip(tm_2).collect();

        // Chimney
        let o2_o5_value = co_gases_2 * 16. / 28.;
        let o2_o5 = if o2_o5_value > o2_others {
            o2_others
        } else {
            o2_o5_value
        };
        let co_o5 = o2_o5 * 1.75;
        json_manager.co_o5_red = co_o5;
        let co2_o5 = o2_o5 * 2.75;

        let co2_tm = co2_gases_2 + co2_o5;
        let co_tm = co_gases_2 - co_o5;
        let n2_tm = n2_gases_2 + n2_others;
        let so2_tm = so2_gases_2;
        let h2o_tm = h2o_gases_2
            + json_manager.obj_water_consumption_red / 1000. * reduction_time
            + h2o_others;
        let o2_tm = o2_gases_2 + o2_others - o2_o5;

        let mut tm_3 = vec![co2_tm, co_tm, n2_tm, so2_tm, h2o_tm, o2_tm];
        let total = tm_3.iter().sum::<Number>();
        tm_3.push(total);

        let co2_nm3_h = co2_tm * 1000. * 22.4 / 44. / reduction_time;
        let co_nm3_h = co_tm * 1000. * 22.4 / 28. / reduction_time;
        let n2_nm3_h = n2_tm * 1000. * 22.4 / 28. / reduction_time;
        let so2_nm3_h = so2_tm * 1000. * 22.4 / 64. / reduction_time;
        let h2o_nm3_h = h2o_tm * 1000. * 22.4 / 18. / reduction_time;
        let o2_nm3_h = o2_tm * 1000. * 22.4 / 32. / reduction_time;

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

impl Debug for ReductionMatter {
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
            .field("\ndf_gases_chimney", &self.df_gases_chimney);

        for (index, reaction) in self.reactions.iter().enumerate() {
            debug_struct.field(&format!("\nreaction[{}]", index + 1), reaction);
        }

        debug_struct.finish()
    }
}
