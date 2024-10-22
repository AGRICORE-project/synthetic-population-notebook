import pandas as pd
import os



def italian_name_conversion(base_path, use_case, year, table_I, crop_codes, totals_variables, weights_var_default="weight_reg"):
    

    USE_CASE = use_case
    YEAR = year

    use_case_abrev = "AND" if USE_CASE.startswith("andalusia") else "ITA" if USE_CASE=="italy" else "POL" if USE_CASE=="poland" else "ESP" if USE_CASE=="spain" else None

    # PATHS ----------------------------------------------------------------------------------------------------------------------------
    #BASE_PATH = os.path.join(PATH_DIFFERENCE, f'./data/use_case_{self.USE_CASE}')
    BASE_PATH = base_path
    print(f"base_path: {BASE_PATH}")

    METADATA_PATH = os.path.join(BASE_PATH, "metadata")
    MICRODATA_PATH = os.path.join(BASE_PATH, "microdata")
    
    # Filenames 
    CATEGORICALS_FILENAME = "categorical_variables.csv"
    MICRODATA_FILENAME = f"{use_case_abrev}{YEAR}.csv"
    MICRODATA_FILE = os.path.join(MICRODATA_PATH, MICRODATA_FILENAME)

    CATEGORICALS_PATH = os.path.join(METADATA_PATH, CATEGORICALS_FILENAME)

    # LOAD INPUT FILES ---------------------------------------------------------------------------------------------------------------------
    # Load microdata
    data = pd.read_csv(MICRODATA_FILE)
    

    print("\n\n\n\n\n\n\n")
    print(weights_var_default in data.columns)
    print("\n\n\n\n\n\n\n")

    # Number of records in the sample
    n_rows = data.shape[0]
    print(f'Number of farms in the sample: {n_rows}')

    # Load categorical variables
    categorical_variables = pd.read_csv(CATEGORICALS_PATH)["Name"].tolist()

    categoricals_final_list = []
    
    # Expand categorical variables
    for cat in categorical_variables:
        
        if cat in table_I:
            for code in crop_codes:
                categoricals_final_list.append(f"{code}.{cat}")
                print("CAT", cat)
                
        else:
            categoricals_final_list.append(cat)
            print("CAT", cat)
    
    #-----------------------------------------------------------------------------------------------------------------

    # Remove ABM from names
    data = data.rename(columns={c: c.replace("ABM_", "") for c in data.columns})

    # Change subsidy by value
    data = data.rename(columns={c: c.replace("subsidy", "value") for c in data.columns})

    # Update variable costs differentating between crops and livestock
    for var in data.columns:
        if var.endswith(".variableCosts"):
            # Animal costs
            if var in ["DAIRY.variableCosts", "OTHER_LIVESTOCK.variableCosts"]:
                data = data.rename(columns={var: var.replace("variableCosts", "variableCostsAnimals")})
            # Crop costs
            else:
                data = data.rename(columns={var: var.replace("variableCosts", "variableCostsCrops")})
                

    # Direct renaming
    data = data.rename(columns={"region_level_3_name": "regionLevel3Name", 
                                "region_level_3": "regionLevel3", 
                                "region_level_1": "regionLevel1", 
                                "region_level_1_name": "regionLevel1Name", 
                                "region_level_2": "regionLevel2", 
                                "region_level_2_name": "regionLevel2Name", 
                                "Anno": "yearNumber", 
                                "Cod_Azienda": "farmCode", 
                                #"OTE": "A_TY_90_TF", 
                                "Genere": "holderGender",
                                "SAU": "agriculturalLandArea", 
                                "SAU_Proprietà": "rentBalanceIn", #"B_UO_10_A", 
                                "SAU_Affitto": "B_UT_20_A", 
                                "Superficie_Forestale": "forestLandArea", 
                                "Classe_Altre_Att_Lucrative": "A_CL_140_C", 
                                })
    
    # Group variables whose meaning is unknown
    UNKNOWN = ["ZSVA", 
                "Cod_Zona_Altimetrica_3", 
                "Zona_Altimetrica_3", 
                "Zona_Altimetrica_5", 
                "Cod_Zona_Altimetrica_5", 
                "Cod_Reg_Agraria", 
                "Regione_Agraria", 
                "OTE", 
                "ID_PoloOTE", 
                "PoloOTE", 
                "UDE_INEA", 
                "UDE", 
                "UDE10", 
                "UDE_EU", 
                "Gruppo_DE", 
                "Produzione_Standard_Aziendale", 
                "Cod_Conduzione", 
                "Conduzione", 
                "Forma_Giuridica", 
                "Cod_Forma_Giuridica",  
                
                
                "Cod_Insediamento", 
                "Insediamento", 
                "Giovane", 
                "Diversificata", 
                "Biologica", 
                "Num_Corpi_Aziendali", 
                "Num_Centri_Aziendali", 
                
                "SAU_Comodato", 
                "SAU_Irrigata", 
                
                "SAU_Comodato", 
                "SAU_Irrigata", 
                "UBA_Totale", 
                "KW_Macchine", 
                "Ore_Totali", 
                
                "UL", 
                "ULF", 
                
                "Cod_Dim_Economica_BDR", 
                "Dim_Economica_BDR", 
                "Unnamed: 63", 
                ]

    # Remove unnecesary and unknown variables from microdata
    data = data.drop(columns=["COD_NUTS3", # COD_NUTS3 = regionLevel2
                            "Cod_Provincia", #
                            "Provincia", # Provincia = reguionLevel2Name
                            "Sigla_Prov", #
                            "Regione", # Regione = regionLevel1Name
                            "Cod_Regione_ISTAT", #
                            "Cod_Regione_UE", 
                            "Cod_Regione_UE", 
                            "Cod_Regione_UE", 
                            "Sigla_Ripartizione", 
                            "Sigla_Ripartizione", 
                            "COD_NUTS2", # COD_NUTS2 = reginoLevel1
                            #"ZSVA", ???????
                            #"Cod_Zona_Altimetrica_3", 
                            #"Zona_Altimetrica_3", 
                            #"Zona_Altimetrica_5", 
                            #"Cod_Zona_Altimetrica_5", 
                            #"Cod_Reg_Agraria", 
                            #"Regione_Agraria", 
                            "machinery", 
                            "Superficie_Totale", # Duplicated
                            "Costo_Opp_Lavoro_Uomo_Orario", # Hourly human cost
                            "Costo_Opp_Lavoro_Macchina_Orario", # Hourly machine cost

                            "Cod_Profilo_Strategico", 
                            "Profilo_Strategico", 
                            "Cod_Classe_di_SAU", 
                            "Classe_di_SAU", 
                            "Incidenza_Altre_Att_Lucrative", 
                            "Cod_Polo_BDR", 
                            "Descrizione_Polo_BDR", 
                            
                            "Cod_INEA", 
                            "Cod_ISTAT", 
                            "Campo_di_osservazione_RICA", 
                            "Cod_Indagine", 
                            "Tipo_Indagine", 
                            "STRATO_ITA_REG", 
                            "STRATO_ITA_OTE", 
                            "STRATO_REG_DE5", 
                            "STRATO_REG_OTE", 
                            "PESO_ITA_REG", 
                            "PESO_ITA_OTE", 
                            "PESO_REG_DE5", 
                            "PESO_REG_OTE", 
                            "FLAG_OUTLIER_ITA_REG", 
                            "FLAG_OUTLIER_ITA_OTE", 
                            "FLAG_OUTLIER_REG_DE5", 
                            "FLAG_OUTLIER_REG_OTE", 
                            "AZIENDE_ITA_OTE", 
                            "AZIENDE_REG_DE5", 
                            "AZIENDE_REG_OTE", 
                            "Unnamed: 22", ] + UNKNOWN)
    
    # Filter by year
    data = data[data["yearNumber"]==int(YEAR)]

    # Fill nans
    data = data.fillna(0)

    # Select Emlia Romagna data
    data = data[data["regionLevel1"]=="ITH5"]

    # Get crop aggregations
    crop_aggregations = [c.replace(".cultivatedArea", "") for c in data.columns if c.endswith(".cultivatedArea")]

    # Check if organicProductionType is available on microdata
    for agg in crop_aggregations:
        if not f"{agg}.organicProductionType" in data.columns:
            organicProductionType_value = "organic" if agg.startswith("ORG") else "conventional"
            data[f"{agg}.organicProductionType"] = organicProductionType_value

        else:
            pass
        
    # Remove duplicated columns
    data = data.loc[:,~data.columns.duplicated()].copy(deep=True)
    # Standarsise column types
    for c in data.columns:
        if c not in categoricals_final_list:
            if str(data[c].dtype)=="object":
                print(c)
                data[c] = data[c].apply(lambda x: x.replace(",", ".")).astype(float)
    
    # Preprocess holderAge
    data["holderAge"] = data["holderAge"].apply(lambda x: int(float(x)))

    # Preprocess holderGender
    data["holderGender"] = data["holderGender"].apply(lambda x: 1 if x=="M" else 2)

    data = data.reset_index(drop=True)

    return data, categoricals_final_list, data[weights_var_default]
