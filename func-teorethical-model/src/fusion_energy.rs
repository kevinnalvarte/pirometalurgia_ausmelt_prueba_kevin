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
use crate::Number;
use std::collections::HashMap;
use std::error::Error;

#[derive(Debug)]
pub struct FusionEnergy {
    pub total_heat_input: Number,
    pub hm_heat_input: HashMap<String, Number>,
    pub total_heat_output: Number,
    pub hm_total_heat_output: HashMap<String, Number>,
    pub total_input_heat_post_combustion: Number,
    pub hm_input_heat_post_combustion: HashMap<String, Number>,
    pub total_output_heat_post_combustion: Number,
    pub hm_output_heat_post_combustion: HashMap<String, Number>,
    pub total_input_heat_gas_cooler: Number,
    pub total_output_heat_gas_cooler: Number,
}

impl OptimizationHelpers for FusionEnergy {}
impl FusionEnergy {
    pub fn new(
        json_manager: &mut JSONManager,
        fusion_matter: &FusionMatter,
    ) -> Result<Self, Box<dyn Error>> {
        let (total_heat_input, hm_heat_input) =
            Self::create_total_heat_input(json_manager, fusion_matter)?;
        let (total_heat_output, hm_total_heat_output) =
            Self::create_total_heat_output(json_manager, fusion_matter)?;
        let (total_input_heat_post_combustion, hm_input_heat_post_combustion) =
            Self::create_total_input_heat_post_combustion(
                json_manager,
                fusion_matter,
                &hm_total_heat_output,
            )?;

        let (total_output_heat_post_combustion, hm_output_heat_post_combustion) =
            Self::create_total_output_heat_post_combustion(
                json_manager,
                fusion_matter,
                total_input_heat_post_combustion,
            )?;

        let (total_input_heat_gas_cooler, total_output_heat_gas_cooler) =
            Self::create_total_heat_gas_cooler(
                json_manager,
                fusion_matter,
                hm_output_heat_post_combustion["mcal_gases_heat_sense"],
                hm_output_heat_post_combustion["mcal_fumes_heat_sense"],
            )?;

        Ok(FusionEnergy {
            total_heat_input,
            hm_heat_input,
            total_heat_output,
            hm_total_heat_output,
            total_input_heat_post_combustion,
            hm_input_heat_post_combustion,
            total_output_heat_post_combustion,
            hm_output_heat_post_combustion,
            total_input_heat_gas_cooler,
            total_output_heat_gas_cooler,
        })
    }

    fn create_total_heat_input(
        json_manager: &JSONManager,
        fusion_matter: &FusionMatter,
    ) -> Result<(Number, HashMap<String, Number>), Box<dyn Error>> {
        let df_f_p = &json_manager.df_fuel_parameters;
        let pm = df_f_p.get_specific_value("pm", 0)?;
        let df_combustion_income = &fusion_matter.df_combustion_income;
        let df_income_charge = &fusion_matter.df_income_charge;
        let petroleum_combustion_kcal_kg = df_f_p.get_specific_value("kcal/m3", 0)?;
        let fuel_combustion =
            df_combustion_income.get_conditional_value("components", "total", "combustible")?;

        // 1. Petroleum Combustion
        let mcal_petroleum_fuel: Number =
            fuel_combustion * 1000. * (petroleum_combustion_kcal_kg * 22.4 / pm) / 1000.;

        let sensible_petroleum_heat_j_kg_k: Number = 1700.;
        let sensible_petroleum_temperature: Number = 105.;

        // 2. Sensible Heat Petroleum
        let mcal_sensible_petroleum: Number = sensible_petroleum_heat_j_kg_k / 4.186
            * fuel_combustion
            * (sensible_petroleum_temperature + 273.)
            / 1000.;

        let sensible_dump_heat_temperature = json_manager
            .theoretical_input
            .temperatura_inicial_escoria_red;

        // 3. Sensible heat initial dump
        let comp_dump_initial_sn =
            json_manager.theoretical_input.otras_configuraciones["composicion_escoria_inicial_sn"];
        let comp_dump_initial_fe =
            json_manager.theoretical_input.otras_configuraciones["composicion_escoria_inicial_fe"];
        let comp_dump_initial_sio2 = json_manager.theoretical_input.otras_configuraciones
            ["composicion_escoria_inicial_sio2"];
        let comp_dump_initial_al2o3 = json_manager.theoretical_input.otras_configuraciones
            ["composicion_escoria_inicial_al2o3"];
        let comp_dump_initial_cao =
            json_manager.theoretical_input.otras_configuraciones["composicion_escoria_inicial_cao"];
        let comp_dump_initial_mgo =
            json_manager.theoretical_input.otras_configuraciones["composicion_escoria_inicial_mgo"];
        let comp_dump_initial_others = 100.
            - (comp_dump_initial_sn
                + comp_dump_initial_fe
                + comp_dump_initial_sio2
                + comp_dump_initial_al2o3
                + comp_dump_initial_cao
                + comp_dump_initial_mgo);
        let initial_weight_dump = json_manager.initial_weight_dump;

        let sno_3_tm: Number =
            comp_dump_initial_sn * initial_weight_dump / 100. * (118.7 + 16.) / 118.7;
        let feo_3_tm: Number = comp_dump_initial_fe * initial_weight_dump / 100. * 71.85 / 55.85;
        let sio2_3_tm: Number = comp_dump_initial_sio2 * initial_weight_dump / 100.;
        let al2o3_3_tm: Number = comp_dump_initial_al2o3 * initial_weight_dump / 100.;
        let cao_3_tm: Number = comp_dump_initial_cao * initial_weight_dump / 100.;
        let mgo_3_tm: Number = comp_dump_initial_mgo * initial_weight_dump / 100.;
        let others_3_tm: Number = comp_dump_initial_others * initial_weight_dump / 100.
            - sno_3_tm * 16. / (118.7 + 16.)
            - feo_3_tm * 16. / 71.85;
        let mut tm_sense_dump_heat = HashMap::new();
        tm_sense_dump_heat.insert("sno", sno_3_tm);
        tm_sense_dump_heat.insert("feo", feo_3_tm);
        tm_sense_dump_heat.insert("sio2", sio2_3_tm);
        tm_sense_dump_heat.insert("al2o3", al2o3_3_tm);
        tm_sense_dump_heat.insert("cao", cao_3_tm);
        tm_sense_dump_heat.insert("mgo", mgo_3_tm);
        tm_sense_dump_heat.insert("otros", others_3_tm);

        let sense_heat_elements = vec!["sno", "feo", "sio2", "al2o3", "cao", "mgo", "otros"];
        let df_energy = &json_manager.df_energy_elements;

        let hm_sense_dump_calories = Self::create_calories_table(
            df_energy,
            &sense_heat_elements,
            sensible_dump_heat_temperature + 273.,
        )?;

        let mut result_sense_heat_init_dump = HashMap::new();
        let mut mcal_sense_init_dump: Number = 0.;
        for element in sense_heat_elements {
            let temp_value = hm_sense_dump_calories[element] * 1000. * tm_sense_dump_heat[element];
            result_sense_heat_init_dump.insert(element.to_string(), temp_value);
            mcal_sense_init_dump += temp_value;
        }

        if let Some(value) = result_sense_heat_init_dump.remove("otros") {
            result_sense_heat_init_dump.insert("others".to_string(), value);
        }
        mcal_sense_init_dump /= 1000.;

        // 4. Sense air heat
        // TODO: Hardcode
        let sense_air_temperature: Number = 31.;
        let hm_air_combustion_income =
            df_combustion_income.generate_hashmap_str_number("components", "air")?;
        let hm_a_air_combustion_income =
            df_combustion_income.generate_hashmap_str_number("components", "a_atomized")?;
        let o2_inf = json_manager.air_o2_fus;
        let n2_inf = json_manager.air_n2_fus;
        let h2o_inf = json_manager.air_h2o_fus;
        let air_post_combustion_fus = json_manager.air_post_combustion_fus;
        let air_infiltration_fus = json_manager.air_infiltration_fus;
        let lance_reactions_fus =
            json_manager.theoretical_input.otras_configuraciones["lanza_reacciona_fusion"];
        let o2_4_tm = hm_air_combustion_income["o"]
            + hm_a_air_combustion_income["o"]
            + o2_inf * air_post_combustion_fus / air_infiltration_fus * lance_reactions_fus;
        let n2_4_tm = hm_air_combustion_income["n"]
            + hm_a_air_combustion_income["n"]
            + n2_inf * air_post_combustion_fus / air_infiltration_fus * lance_reactions_fus;
        let h2o_4_tm = hm_air_combustion_income["h2o"]
            + hm_a_air_combustion_income["h2o"]
            + h2o_inf * air_post_combustion_fus / air_infiltration_fus * lance_reactions_fus;
        let sense_air_elements = vec!["o2", "n2", "h2o_e"];
        let mut tm_sense_air = HashMap::new();
        tm_sense_air.insert("o2", o2_4_tm);
        tm_sense_air.insert("n2", n2_4_tm);
        tm_sense_air.insert("h2o_e", h2o_4_tm);
        let hm_sense_air_calories = Self::create_calories_table(
            df_energy,
            &sense_air_elements,
            sense_air_temperature + 273.,
        )?;
        let mut mcal_sense_air: Number = 0.;
        let mut result_sense_air = HashMap::new();
        for element in sense_air_elements {
            let temp_value = hm_sense_air_calories[element] * 1000. * tm_sense_air[element];
            result_sense_air.insert(element.to_string(), temp_value);
            mcal_sense_air += temp_value;
        }
        mcal_sense_air /= 1000.;

        // 5. Heat Sense Oxygen
        // TODO: Hardcode
        let sense_oxygen_temperature: Number = 27.;
        let o2_5_tm = df_combustion_income.get_conditional_value("components", "o", "oxygen")?;
        let n2_5_tm = df_combustion_income.get_conditional_value("components", "n", "oxygen")?;
        let h2o_5_tm = df_combustion_income.get_conditional_value("components", "h2o", "oxygen")?;
        let sense_oxygen_elements = vec!["o2", "n2", "h2o_e"];

        let hm_sense_oxygen_calories = Self::create_calories_table(
            df_energy,
            &sense_oxygen_elements,
            sense_oxygen_temperature + 273.,
        )?;
        let mut mcal_sense_oxygen: Number = 0.;
        let mut result_sense_oxygen = HashMap::new();
        let mut tm_sense_oxygen = HashMap::new();
        tm_sense_oxygen.insert("o2", o2_5_tm);
        tm_sense_oxygen.insert("n2", n2_5_tm);
        tm_sense_oxygen.insert("h2o_e", h2o_5_tm);

        for element in sense_oxygen_elements {
            let temp_value = hm_sense_oxygen_calories[element] * 1000. * tm_sense_oxygen[element];
            result_sense_oxygen.insert(element.to_string(), temp_value);
            mcal_sense_oxygen += temp_value;
        }
        mcal_sense_oxygen /= 1000.;

        // Chemical exo reactions
        let elements = vec![
            "cu2s + sn = sns + 2 cu",
            "s + sn = sns",
            "c2h6 + 3.5 o2 = 2 co2 + 3 h2o",
            "fe + 0.5 o2 = feo",
            "sn + 0.5 o2 = sno",
            "co + 0.5 o2 = co2",
            "c + 0.5 o2  = co",
            "sno + 0.5 o2 = sno2",
            "sns + 2 o2 = sno2 + so2",
            "0.5 s2 + o2 = so2",
            "c2h6 + 3.5 o2 = 2 co2 + 3 h2o",
        ];
        let reactions = &fusion_matter.reactions;
        let hm_reactions_kcal = json_manager
            .df_energy_equations
            .generate_hashmap_str_number("element", "kcal/kg")?;
        let mut hm_reactions_1 = HashMap::new();
        let mut hm_reactions_2 = HashMap::new();

        // TODO Review duplicated reaction of co
        hm_reactions_1.insert("cu2s + sn = sns + 2 cu", reactions[14]["cu2s"]);
        hm_reactions_1.insert("s + sn = sns", reactions[15]["s"]);
        hm_reactions_1.insert("c2h6 + 3.5 o2 = 2 co2 + 3 h2o", reactions[16]["c2h6"]);
        hm_reactions_1.insert("fe + 0.5 o2 = feo", reactions[3]["fe"]);
        hm_reactions_1.insert("sn + 0.5 o2 = sno", reactions[4]["sn"]);
        hm_reactions_1.insert("co + 0.5 o2 = co2", reactions[17]["co"]);
        hm_reactions_1.insert("c + 0.5 o2  = co", reactions[18]["c"]);

        hm_reactions_2.insert("sno + 0.5 o2 = sno2", reactions[24]["sno"]);
        hm_reactions_2.insert("sns + 2 o2 = sno2 + so2", reactions[25]["sns"]);
        hm_reactions_2.insert("co + 0.5 o2 = co2", reactions[26]["co"]);
        hm_reactions_2.insert("0.5 s2 + o2 = so2", reactions[27]["0.5s2"]);
        hm_reactions_2.insert(
            "c2h6 + 3.5 o2 = 2 co2 + 3 h2o",
            (df_income_charge.get_conditional_value("molecules", "c2h6", "total")?
                - fusion_matter.reactions[16]["c2h6"])
                * lance_reactions_fus,
        );

        let mut mcal_reactions = 0.;
        for element in elements {
            let r_1 =
                hm_reactions_1.get(element).unwrap_or(&0.0) * hm_reactions_kcal[element] * -1000.;
            let r_2 =
                hm_reactions_2.get(element).unwrap_or(&0.0) * hm_reactions_kcal[element] * -1000.;
            mcal_reactions += r_1 + r_2;
        }
        mcal_reactions /= 1000.;

        let result = mcal_reactions
            + mcal_sense_oxygen
            + mcal_sense_air
            + mcal_sensible_petroleum
            + mcal_sense_init_dump
            + mcal_petroleum_fuel;
        let mut hm_heat_input = HashMap::new();
        hm_heat_input.insert("mcal_reactions".to_string(), mcal_reactions);
        hm_heat_input.insert("mcal_sense_oxygen".to_string(), mcal_sense_oxygen);
        hm_heat_input.insert("mcal_sense_air".to_string(), mcal_sense_air);
        hm_heat_input.insert(
            "mcal_sensible_petroleum".to_string(),
            mcal_sensible_petroleum,
        );
        hm_heat_input.insert("mcal_sense_init_dump".to_string(), mcal_sense_init_dump);
        hm_heat_input.insert("mcal_petroleum_fuel".to_string(), mcal_petroleum_fuel);
        Ok((result, hm_heat_input))
    }

    fn create_total_heat_output(
        json_manager: &JSONManager,
        fusion_matter: &FusionMatter,
    ) -> Result<(Number, HashMap<String, Number>), Box<dyn Error>> {
        // 1 Metal Sense Heat
        let hm_metal = fusion_matter
            .df_metal
            .generate_hashmap_str_number("elements", "tm")?;
        let elements_1 = vec!["sn", "fe", "pb", "sb", "as", "cu", "otros"];
        let stove_energy_fusion = json_manager.obj_stove_energy_fus;
        let hm_1_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_1,
            stove_energy_fusion + 273.,
        )?;
        let mut mcal_metal_heat_sense = 0.;
        for element in elements_1 {
            if element == "otros" {
                mcal_metal_heat_sense += hm_metal["others"] * 1000. * hm_1_calories["otros"];
            } else {
                mcal_metal_heat_sense += hm_metal[element] * 1000. * hm_1_calories[element];
            }
        }

        mcal_metal_heat_sense /= 1000.;

        // 2. Dump Heat Sense
        let hm_dump = fusion_matter
            .df_dump
            .generate_hashmap_str_number("elements", "tm")?;
        let elements_2 = vec!["sno", "feo", "sio2", "al2o3", "cao", "mgo", "otros"];
        let hm_2_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_2,
            stove_energy_fusion + 273.,
        )?;
        let mut mcal_dump_heat_sense = 0.;
        for element in elements_2 {
            if element == "otros" {
                mcal_dump_heat_sense += hm_dump["others"] * 1000. * hm_2_calories["otros"];
            } else {
                mcal_dump_heat_sense += hm_dump[element] * 1000. * hm_2_calories[element];
            }
        }

        mcal_dump_heat_sense /= 1000.;

        // 3. Gas Heat Sense
        let hm_gases = &fusion_matter.hm_gases;
        let elements_3 = vec!["co2", "co", "n2", "so2", "h2o", "ch4", "o2", "s2"];
        let hm_3_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_3,
            stove_energy_fusion + 200. + 273.,
        )?;

        let mut mcal_gases_heat_sense = 0.;
        for element in elements_3 {
            if element == "ch4" {
                mcal_gases_heat_sense += hm_gases["c2h6"] * 1000. * hm_3_calories["ch4"];
            } else {
                mcal_gases_heat_sense += hm_gases[element] * 1000. * hm_3_calories[element];
            }
        }

        mcal_gases_heat_sense /= 1000.;

        // 4. Dross Fe Heat Sense
        let hm_dross = fusion_matter
            .df_dross_fe
            .generate_hashmap_str_number("elements", "tm")?;

        let elements_4 = vec!["sn", "fe", "pb", "sb", "as", "cu", "otros"];
        let hm_4_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_4,
            stove_energy_fusion + 273.,
        )?;
        let mut mcal_dross_heat_sense = 0.;
        for element in elements_4 {
            if element == "otros" {
                mcal_dross_heat_sense += hm_dross["others"] * 1000. * hm_4_calories["otros"];
            } else {
                mcal_dross_heat_sense += hm_dross[element] * 1000. * hm_4_calories[element];
            }
        }

        mcal_dross_heat_sense /= 1000.;

        // 5. Fumes Heat Sense
        let hm_fumes = fusion_matter
            .df_fumes
            .generate_hashmap_str_number("elements", "tm")?;
        let elements_5 = vec![
            "sno v", "sno2", "sns v", "feo", "pb", "sb", "as", "cu", "otros",
        ];
        let hm_5_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_5,
            stove_energy_fusion + 200. + 273.,
        )?;
        let mut mcal_fumes_heat_sense = 0.;
        for element in elements_5 {
            if element == "otros" {
                mcal_fumes_heat_sense += hm_fumes["others"] * 1000. * hm_5_calories["otros"];
            } else if element == "sno v" {
                mcal_fumes_heat_sense += hm_fumes["sno"] * 1000. * hm_5_calories["sno v"];
            } else if element == "sns v" {
                mcal_fumes_heat_sense += hm_fumes["sns"] * 1000. * hm_5_calories["sns v"];
            } else {
                mcal_fumes_heat_sense += hm_fumes[element] * 1000. * hm_5_calories[element];
            }
        }
        mcal_fumes_heat_sense /= 1000.;

        // 6. Reactions
        let mut hm_reactions_6 = HashMap::new();
        let reactions = &fusion_matter.reactions;
        let reactions_calories = &json_manager
            .df_energy_equations
            .generate_hashmap_str_number("element", "kcal/kg")?;
        let reactions_list = vec![
            "sno2 + c = sno + co",
            "fes2 + sno = sns + feo + 0.5 s2",
            "fesn2   = 2 sn + fe",
            "fe + sno = feo + sn",
            "sno + c = sn + co",
            "fe2o3 + c = 2 feo + co",
            "feo + c = fe + co",
            "caco3  = cao + co2",
            "mgco3  = mgo + co2",
        ];
        hm_reactions_6.insert("sno2 + c = sno + co", reactions[0]["sno2"]);
        hm_reactions_6.insert("fes2 + sno = sns + feo + 0.5 s2", reactions[1]["fes2"]);
        hm_reactions_6.insert("fesn2   = 2 sn + fe", reactions[2]["fesn2"]);
        hm_reactions_6.insert("fe + sno = feo + sn", reactions[5]["fe"]);
        hm_reactions_6.insert("sno + c = sn + co", reactions[6]["sno"]);
        hm_reactions_6.insert("fe2o3 + c = 2 feo + co", reactions[9]["fe2o3"]);
        hm_reactions_6.insert("feo + c = fe + co", reactions[10]["feo"]);
        hm_reactions_6.insert("caco3  = cao + co2", reactions[12]["caco3"]);
        hm_reactions_6.insert("mgco3  = mgo + co2", reactions[13]["mgco3"]);

        let mut mcal_reactions = 0.;
        for reaction in reactions_list {
            mcal_reactions += hm_reactions_6[reaction] * 1000. * reactions_calories[reaction]
        }
        mcal_reactions /= 1000.;

        // 7. Latent Heat
        let hm_income_charge = fusion_matter
            .df_income_charge
            .generate_hashmap_str_number("molecules", "total")?;
        let hm_water_calories = &json_manager
            .df_energy_water
            .generate_hashmap_str_number("element", "kcal/kg")?;

        let water_7_tm = hm_income_charge["h2o"];
        let sn_7_tm = hm_income_charge["sn"] + reactions[2]["2sn"];
        let sno_7_tm = reactions[7]["sno(v)"];
        let sns_7_tm = fusion_matter
            .df_fumes
            .get_conditional_value("elements", "sns", "tm")?;

        let mcal_latent_heat: Number = water_7_tm * hm_water_calories["evaporacion agua"]
            + sn_7_tm * hm_water_calories["sn(s) = sn(l)"]
            + sno_7_tm * hm_water_calories["sno(s) = sno(g)"]
            + sns_7_tm * hm_water_calories["sns(s) = sns(g)"];

        // 8. Refrigeration heat

        let water_flow_8 = json_manager.theoretical_input.otras_configuraciones["flujo_agua"];
        let initial_temp_8 = json_manager
            .theoretical_input
            .temperatura_inicial_fusion_horno;
        let final_temp_8 = json_manager
            .theoretical_input
            .temperatura_final_fusion_horno;

        let mcal_refrigeration = water_flow_8 * 1000. * (final_temp_8 - initial_temp_8) / 1000.
            * json_manager.fusion_time;
        let lost = json_manager.theoretical_input.otras_configuraciones["perdidas_fusion"];
        // 9. Lost Heat
        let mcal_heat_lost = lost * json_manager.fusion_time;

        let mut hm_result = HashMap::new();
        hm_result.insert("mcal_metal_heat_sense".to_string(), mcal_metal_heat_sense);
        hm_result.insert("mcal_dump_heat_sense".to_string(), mcal_dump_heat_sense);
        hm_result.insert("mcal_gases_heat_sense".to_string(), mcal_gases_heat_sense);
        hm_result.insert("mcal_dross_heat_sense".to_string(), mcal_dross_heat_sense);
        hm_result.insert("mcal_fumes_heat_sense".to_string(), mcal_fumes_heat_sense);
        hm_result.insert("mcal_refrigeration".to_string(), mcal_refrigeration);
        hm_result.insert("mcal_reactions".to_string(), mcal_reactions);
        hm_result.insert("mcal_heat_lost".to_string(), mcal_heat_lost);
        hm_result.insert("mcal_latent_heat".to_string(), mcal_latent_heat);

        let result = mcal_metal_heat_sense
            + mcal_dump_heat_sense
            + mcal_gases_heat_sense
            + mcal_dross_heat_sense
            + mcal_fumes_heat_sense
            + mcal_refrigeration
            + mcal_reactions
            + mcal_heat_lost
            + mcal_latent_heat;

        Ok((result, hm_result))
    }

    fn create_total_input_heat_post_combustion(
        json_manager: &JSONManager,
        fusion_matter: &FusionMatter,
        hm_total_heat_output: &HashMap<String, Number>,
    ) -> Result<(Number, HashMap<String, Number>), Box<dyn Error>> {
        // 1. Gases Sense Heat
        let mcal_gases_heat_sense = hm_total_heat_output["mcal_gases_heat_sense"];
        // 2. Fumes Sense Heat
        let mcal_fumes_heat_sense = hm_total_heat_output["mcal_fumes_heat_sense"];
        // 3. Air Sense heat
        // TODO: Hardcode
        let temperature_3 = 31.;
        let o2_inf = json_manager.air_o2_fus;
        let n2_inf = json_manager.air_n2_fus;
        let h2o_inf = json_manager.air_h2o_fus;
        let air_post_combustion_fus = json_manager.air_post_combustion_fus;
        let air_infiltration_fus = json_manager.air_infiltration_fus;
        let reactions = &fusion_matter.reactions;
        let code_reaction_fus =
            json_manager.theoretical_input.otras_configuraciones["codo_reacciona_fusion"];
        let elements = vec!["o2", "n2", "h2o_e"];
        let hm_air_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements,
            temperature_3 + 273.,
        )?;
        let o2_3_tm = o2_inf * air_post_combustion_fus
            / (air_infiltration_fus + air_post_combustion_fus)
            * code_reaction_fus;
        let n2_3_tm = n2_inf * air_post_combustion_fus
            / (air_infiltration_fus + air_post_combustion_fus)
            * code_reaction_fus;
        let h2o_3_tm = h2o_inf * air_post_combustion_fus
            / (air_infiltration_fus + air_post_combustion_fus)
            * code_reaction_fus;

        let mcal_air_heat_sense = o2_3_tm * hm_air_calories["o2"]
            + h2o_3_tm * hm_air_calories["h2o_e"]
            + n2_3_tm * hm_air_calories["n2"];

        // 4. Reactions Heat Sense
        let hm_energy_equations = json_manager
            .df_energy_equations
            .generate_hashmap_str_number("element", "kcal/kg")?;

        let equations_4 = vec![
            "sno + 0.5 o2 = sno2",
            "sns + 2 o2 = sno2 + so2",
            "co + 0.5 o2 = co2",
            "0.5 s2 + o2 = so2",
            "c2h6 + 3.5 o2 = 2 co2 + 3 h2o",
        ];

        let mut hm_4_reactions = HashMap::new();
        hm_4_reactions.insert("sno + 0.5 o2 = sno2", reactions[29]["sno"]);
        hm_4_reactions.insert("sns + 2 o2 = sno2 + so2", reactions[30]["sns"]);
        hm_4_reactions.insert("co + 0.5 o2 = co2", reactions[31]["co"]);
        hm_4_reactions.insert("0.5 s2 + o2 = so2", reactions[32]["0.5s2"]);
        hm_4_reactions.insert("c2h6 + 3.5 o2 = 2 co2 + 3 h2o", reactions[33]["c2h6"]);

        let mut mcal_reactions_heat_sense: Number = 0.;
        for element in equations_4 {
            mcal_reactions_heat_sense += -hm_4_reactions[element] * hm_energy_equations[element];
        }

        let result = mcal_reactions_heat_sense
            + mcal_gases_heat_sense
            + mcal_fumes_heat_sense
            + mcal_air_heat_sense;

        let mut hm_input_heat_post_combustion = HashMap::new();
        hm_input_heat_post_combustion.insert(
            "mcal_reactions_heat_sense".to_string(),
            mcal_reactions_heat_sense,
        );
        hm_input_heat_post_combustion
            .insert("mcal_gases_heat_sense".to_string(), mcal_gases_heat_sense);
        hm_input_heat_post_combustion
            .insert("mcal_fumes_heat_sense".to_string(), mcal_fumes_heat_sense);
        hm_input_heat_post_combustion
            .insert("mcal_air_heat_sense".to_string(), mcal_air_heat_sense);

        Ok((result, hm_input_heat_post_combustion))
    }

    fn create_total_output_heat_post_combustion(
        json_manager: &JSONManager,
        fusion_matter: &FusionMatter,
        total_input_heat_post_combustion: Number,
    ) -> Result<(Number, HashMap<String, Number>), Box<dyn Error>> {
        // 1. Heat Sense Gases
        let hm_gases_post = &fusion_matter.hm_gases_post;
        let temperature_1 = json_manager.obj_post_comb_energy_fus;
        let elements_1 = vec!["co2", "co", "n2", "so2", "h2o", "o2"];
        let hm_1_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_1,
            temperature_1 + 273.,
        )?;

        let mut mcal_gases_heat_sense: Number = 0.;
        for element in elements_1 {
            mcal_gases_heat_sense += hm_gases_post[element] * hm_1_calories[element];
        }

        // 2. Heat Sense Fumes
        let elements_2 = vec!["sno2", "feo", "pb", "sb", "as", "cu", "otros"];

        let hm_2_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_2,
            temperature_1 + 273.,
        )?;
        let hm_post_combustion_fumes = &fusion_matter.hm_post_combustion_fumes;

        let mut mcal_fumes_heat_sense: Number = 0.;
        for element in elements_2 {
            if element == "otros" {
                mcal_fumes_heat_sense +=
                    hm_2_calories[element] * hm_post_combustion_fumes["others"];
            } else {
                mcal_fumes_heat_sense += hm_2_calories[element] * hm_post_combustion_fumes[element];
            }
        }

        // 3. Heat losses
        let lost_1 = json_manager.theoretical_input.otras_configuraciones["post_comb_fus"];
        let mcal_loses = total_input_heat_post_combustion * lost_1 / 100.;

        let result = mcal_loses + mcal_fumes_heat_sense + mcal_gases_heat_sense;
        let mut hm_output_heat_post_combustion = HashMap::new();
        hm_output_heat_post_combustion.insert("mcal_loses".to_string(), mcal_loses);
        hm_output_heat_post_combustion
            .insert("mcal_fumes_heat_sense".to_string(), mcal_fumes_heat_sense);
        hm_output_heat_post_combustion
            .insert("mcal_gases_heat_sense".to_string(), mcal_gases_heat_sense);

        Ok((result, hm_output_heat_post_combustion))
    }

    fn create_total_heat_gas_cooler(
        json_manager: &JSONManager,
        fusion_matter: &FusionMatter,
        mcal_gases_heat_sense: Number,
        mcal_fumes_heat_sense: Number,
    ) -> Result<(Number, Number), Box<dyn Error>> {
        let eq_1 = json_manager.df_energy_equations.get_conditional_value(
            "element",
            "co + 0.5 o2 = co2",
            "kcal/kg",
        )?;
        let mcal_exo_reaction = -eq_1 * json_manager.co_o5_fus;

        let total_input_heat_gas_cooler =
            mcal_exo_reaction + mcal_fumes_heat_sense + mcal_gases_heat_sense;

        // Fall apart
        // 1. Water evaporation
        let water_consumption_fus = json_manager.obj_water_consumption_fus;
        let fusion_time = json_manager.fusion_time;
        let water_evaporation_kcal = json_manager.df_energy_water.get_conditional_value(
            "element",
            "evaporacion agua",
            "kcal/kg",
        )?;
        let mcal_water_evaporation =
            water_consumption_fus * fusion_time / 1000. * water_evaporation_kcal;

        // 2. Heat Sense Gas Cooler
        let hm_gases_chimney = fusion_matter
            .df_gases_chimney
            .generate_hashmap_str_number("elements", "tm")?;
        let elements_2 = vec!["co2", "co", "n2", "so2", "h2o", "o2"];
        let temperature = json_manager
            .theoretical_input
            .temperatura_salida_gas_cooler_fusion;
        let hm_2_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_2,
            temperature + 273.,
        )?;

        let mut mcal_gases_sense: Number = 0.;
        for element in elements_2 {
            mcal_gases_sense += hm_gases_chimney[element] * hm_2_calories[element];
        }

        // 3. Heat Sense Fumes
        let hm_post_combustion_fumes = &fusion_matter.hm_post_combustion_fumes;
        let elements_3 = vec!["sno2", "feo", "pb", "sb", "as", "cu", "otros"];
        let hm_3_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_3,
            temperature + 273.,
        )?;
        let mut mcal_fumes_sense: Number = 0.;
        for element in elements_3 {
            if element == "otros" {
                mcal_fumes_sense += hm_post_combustion_fumes["others"] * hm_3_calories[element];
            } else {
                mcal_fumes_sense += hm_post_combustion_fumes[element] * hm_3_calories[element];
            }
        }
        let gas_cooler_fus = json_manager.theoretical_input.otras_configuraciones["gas_cooler_fus"];

        let mcal_lost_heat = gas_cooler_fus / 100. * total_input_heat_gas_cooler;

        let total_output_heat_gas_cooler =
            mcal_water_evaporation + mcal_gases_sense + mcal_fumes_sense + mcal_lost_heat;

        Ok((total_input_heat_gas_cooler, total_output_heat_gas_cooler))
    }
}
