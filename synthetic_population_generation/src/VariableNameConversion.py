import os
import pandas as pd
import numpy as np
import random
from src.crop_representativeness_computer import RepresentativenessComputer
from src.generate_missing_totals import WeightComputer


class VariableNameConversion():
    def __init__(self, base_path, use_case, year, totals_variables, PATH_DIFFERENCE=""):
        """
        Class to unify the microdata variable names prior to make data transformations and synthetic population generation
        """

        self.USE_CASE = use_case
        self.YEAR = year

        self.totals_variables = totals_variables

        use_case_abrev = "AND" if self.USE_CASE.startswith("andalusia") else \
                         "ITA" if self.USE_CASE=="italy" else \
                         "PL" if self.USE_CASE=="poland" else \
                         "CM" if self.USE_CASE=="greece" else \
                         "TRIAL" if self.USE_CASE=="trial" else "TRIAL" if self.USE_CASE=="trial2" else "ESP" if self.USE_CASE=="spain2" else None
        #"ELL" if self.USE_CASE=="greece" else \
                         
        # PATHS ----------------------------------------------------------------------------------------------------------------------------
        #BASE_PATH = os.path.join(PATH_DIFFERENCE, f'./data/use_case_{self.USE_CASE}')
        BASE_PATH = base_path

        self.METADATA_PATH = os.path.join(BASE_PATH, "metadata")
        MICRODATA_PATH = os.path.join(BASE_PATH, "microdata")
        self.RESULTS_PATH = os.path.join(BASE_PATH, "results")
        if not "results" in os.listdir(BASE_PATH):
            os.mkdir(self.RESULTS_PATH)

        # Filenames 
        CROPS_CODES_FILENAME = "Product_Mapping.csv"
        TOTALS_FILENAME = f'totals_{self.YEAR}.csv'
        CATEGORICALS_FILENAME = "categorical_variables.csv"
        MICRODATA_FILENAME = f"{use_case_abrev}{self.YEAR}.csv"
        MICRODATA_FILE = os.path.join(MICRODATA_PATH, MICRODATA_FILENAME)

        self.TOTALS_PATH = os.path.join(self.METADATA_PATH, TOTALS_FILENAME)
        CATEGORICALS_PATH = os.path.join(self.METADATA_PATH, CATEGORICALS_FILENAME)

        # LOAD INPUT FILES ---------------------------------------------------------------------------------------------------------------------
        # Load microdata
        
        self.microdata = pd.read_csv(MICRODATA_FILE)
        
        # Filter microdata by NUTS3 value
        #self.microdata = self.microdata[self.microdata["A_LO_40_N"].isin(self.NUTS3)].reset_index(drop=True)

        # Number of records in the sample
        self.n_rows = self.microdata.shape[0]
        print(f'Number of farms in the sample: {self.n_rows}')

        # Load categorical variables
        self.categorical_variables = pd.read_csv(CATEGORICALS_PATH)["Name"].tolist()
        
        # Remove empty columns 
        # Columns with none value available will appear as missing variables
        self.microdata = self.microdata.loc[:, self.microdata.notna().any()]

        # Load FADN codes definition
        self.crop_codes = pd.read_csv(os.path.join(self.METADATA_PATH, CROPS_CODES_FILENAME))
        
        self.crop_codes = self.crop_codes.drop_duplicates()
        
        # Split crop codes into conventional and organic
        self.animal_codes = self._transform_table_codes(pd.read_csv(os.path.join(self.METADATA_PATH, "animal_codes.csv")))
        self.animal_products_codes = self._transform_table_codes(pd.read_csv(os.path.join(self.METADATA_PATH, "animal_products_codes.csv")))

        #self.subsidies_codes = self._transform_table_codes_policies(pd.read_csv(os.path.join(self.METADATA_PATH, "policies.csv")))
        self.subsidies_codes = pd.read_csv(os.path.join(self.METADATA_PATH, "subsidies.csv"))

        # Create some synthetic columns 
        # Milk production costs
        for fadn, agri in zip(["H_OS_4030_V", "H_OS_4040_V", "H_OS_4050_V", "H_OS_4060_V", ], ["261.milkVariableCosts", "262.milkVariableCosts", "311.milkVariableCosts", "321.milkVariableCosts", ]):
            if fadn in self.microdata.columns:
                self.microdata = self.microdata.rename(columns={fadn: agri})
        # Dairy cows
        for fadn, agri in zip(["J_AN_261_A", "J_AN_262_A", "J_AN_263_A", ], ["261.dairyCows", "262.dairyCows", "263.dairyCows"]):
            if fadn in self.microdata.columns:
                self.microdata = self.microdata.rename(columns={fadn: agri})
        # -----------------------------------------------------------------------------------------------------------------------------

        # METHODS ----------------------------------------------------------------------------------------------------------------------
        self.Organic_variables_dict = {
            "organicFarming":        "A_CL_140_C", # Organic farming 
            "sectorsOrganicFarming": "A_CL_141_C", # Sectors in organic farming
        }

        # Record organic-type variables in the data
        self.organic_variables = self._record_organic_type_variables()

        # Get a dataframe with the information requried to compose totals
        self.totals_variables_df = self._get_totals_df()

        # -----------------------------------------------------------------------------------------------------------------------------
        # Required to build organic version of crops although not required in the agents
        table_I_dict = {
            # Defined
            "quantitySold":   ['I_SA_', '_Q'], # Quantity of Sold Production ([tons])
            "valueSales":     ['I_SA_', '_V'], # Value of Sales (PLV - [€])
            "cropProduction": ['I_PR_', '_Q'], # This variable temporarily holds crop production. Later, it will be corrctd using sellingPrice. Value of total production (PLT - [€])
            "irrigatedArea":  ['I_A_', '_IR'], # Irrigated Area (IA - [ha])
            "cultivatedArea": ['I_A_', '_TA'], # Utilized Agricultural Area (UAA - [ha])
            "quantityUsed":   ['I_FU_', '_V'], # This variable holds "value" value in [€]. Later, it will be converted into quantity using sellingPrice. Quantity of Used Production ([tons])
            
            # Not available -> Generated from other data
            "organicProductionType": ["", ".organicProductionType"], 
            "variableCostsCrops":    ["", ".variableCostsCrops"], # Variable Costs per produced unit (CV - [€/ton])
            "landValue":             ["", ".landValue"], # Land Value (PVF - [€])
            "sellingPrice":          ["", ".sellingPrice"], # Unit selling price (PVU - [€/unit])
            }
    
        self.table_I = pd.DataFrame(table_I_dict, index=["start", "end"]).transpose()

        table_J_dict = {
            # Defined
            "numberOfAnimals":              ['J_AN_', '_A'] if self.USE_CASE == "andalusia" else ['J_CV_', '_N'], # Number of Animals [units] 
            "numberOfAnimalsSold":          ['J_SA_', '_N'], # Number of Animals Sold [units]
            "valueSoldAnimals":             ['J_SA_', '_V'], # Value of Sold Animals ([€])
            "numberAnimalsRearingBreading": ['J_SR_', '_N'], # Number of Animals for Rearing/Breeding [units]
            "valueAnimalsRearingBreading":  ['J_SR_', '_V'], # Value of Animals for Rearing/Breeding ([€])
            "numberAnimalsForSlaughtering": ['J_SS_', '_N'], # Number of Animals for Slaughtering [units]
            "valueSlaughteredAnimals":      ['J_SS_', '_V'], # Value of Slaughtered Animals ([€])
        }

        # Convert to dataframe
        self.table_J = pd.DataFrame(table_J_dict, index=["start", "end"]).transpose()

        table_K_milk_dict = {
            # Animal products codes
            # 261 Cows' milk  
            # 262 Buffalo's cows' milk 
            # 311 Sheep's milk 
            # 319 Other sheep
            # 321 Goat's milk
            # Any other code -> VALUE = 0
            "milkTotalProduction": ['K_PR_', '_Q'], # Number of tons of milk produced [tons]
            "milkProductionSold":  ['K_SA_', '_Q'], # Number of tons of milk sold [tons]
            "milkTotalSales":      ['K_SA_', '_V'], # Value of milk sold ([€])

            # Not available
            "milkVariableCosts":    ["", ".milkVariableCosts"],   # Variable Costs per produced unit (CV - [€/ton])
            "dairyCows":            ["", ".dairyCows"],           # Number of dairy cows [UBA - [units]]
            "variableCostsAnimals": ["", ".variableCostsAnimals"] # Average variable cost per unit of product[€/ ton]
        }

        self.milk_codes = [261, 262, 311, 319, 321]

        # Convert to dataframe
        self.table_K_milk = pd.DataFrame(table_K_milk_dict, index=["start", "end"]).transpose()

        table_K_wool_dict = {
            # Wool
            # 330 Wool
            "woolTotalProduction": ['K_PR_', '_Q'], # Wool Production Quantity 
            "woolProductionSold":  ['K_SA_', '_Q'], # Wool Sales Quantity 
        }
        
        self.wool_codes = [330]

        # Convert to dataframe
        self.table_K_wool = pd.DataFrame(table_K_wool_dict, index=["start", "end"]).transpose()

        table_K_eggs_dict = {
            # Eggs
            "eggsTotalProduction": ['K_PR_', '_Q'], # Eggs Production Quantity 
            "eggsProductionSold":  ['K_SA_', '_Q'], # Eggs Sales Quantity 
            "eggsTotalSales":      ['K_SA_', '_V'], # Eggs Sales Value 
        }

        # 531 Eggs for human consumption (all poultry)
        # 532 Eggs for hatching (all poultry)
        self.eggs_codes = [531, 532]

        self.table_K_eggs = pd.DataFrame(table_K_eggs_dict, index=["start", "end"]).transpose()

        table_K_manure_dict = {
            # Manure 
            # 800 Manure
            "manureTotalSales": ['K_SA_', '_V'], #Sales Value 
        }

        self.manure_codes = [800]
        self.table_K_manure = pd.DataFrame(table_K_manure_dict, index=["start", "end"]).transpose()

        table_K_other_dict = {
            # Required
            #"valueSold":    ["K_SA_", "_V"], 
            #"sellingPrice": ["", ".sellingPrice"], # Average shell price per unit of product[€/ ton]
        }

        self.other_codes = [700, 900, 1100, 1120, 1130, 1140, 1150, 1190, 1200]
        self.table_K_other = pd.DataFrame(table_K_other_dict, index=["start", "end"]).transpose()
        
        #------------------------------------------------------------------------------------------------------------
        # Group all animal variables
        self.animal_variables = pd.concat([self.table_J, self.table_K_milk, self.table_K_wool, self.table_K_eggs, self.table_K_manure, self.table_K_other, ])

        #self.milk_codes = [261, 262, 311, 321]
        #self.wool_codes = [330]
        #self.eggs_codes = [531, 532]
        #self.manure_codes = [800]
        #self.other_codes = [700, 900, 1100, 1120, 1130, 1140, 1150, 1190, 1200]
        #-------------------------------------------------------------------------------------------------------------

        # This table 
        table_M_dict = {
            #"policyIdentifier":  ["", ".policyIdentifier"], 
            #"policyDescription": ["", ".policyDescription"], 
            #"isCoupled":         ["", ".isCoupled"], 
            "value":             ["M_S_", "_FI_BU_V"],
        }
        
        self.table_M = pd.DataFrame(table_M_dict, index=["start", "end"]).transpose()

        self.Farm_dict = {
            "lat":      "A_LO_20_DG", 
            "long":     "A_LO_30_DG", 
            "altitude": "A_CL_170_C", 

            "farmCode": "A_ID_10_H", 
            "technicalEconomicOrientation": "A_TY_90_TF", 
            "weight_ra": "A_TY_80_W", 

            "regionLevel1Name": "regionLevel1Name",
            "regionLevel2Name": "regionLevel2Name", 
            "regionLevel3Name": "regionLevel3Name", 
            
            # Not available
            "regionLevel1": "A_LO_40_N2",   # NUTS2
            "regionLevel2": "A_LO_40_N",    # NUTS3
            "regionLevel3": "regionLevel3", # Region

            "weight_reg":     "weight_reg", 
            "rentBalanceIn":  "B_UT_20_A",
            "rentBalanceOut": "I_A_90100_TA", 
        }
        
        #NUTS2
        #NUTS3
        #self.NUTS2 = NUTS2
        #self.NUTS3 = NUTS3

        # Link between regionLevel and regionLevel name
        self.NUTS2_names = {"ES61": "Andalucia"}
        self.NUTS3_names = {"ES611": "Almería", "ES612": "Cádiz","ES613": "Córdoba", "ES614": "Granada", "ES615": "Huelva", "ES616": "Jaén", "ES617": "Málaga", "ES618": "Sevilla"}
        self.REGION_names = {"level3_1": "level3_1name", "level3_2": "level3_2name", "level3_3": "level3_3name", "level3_4": "level3_4name", "level3_5": "level3_5name", "level3_6": "level3_6name", "level3_7": "level3_7name"}
        
        self.ClosingValue_dict = {
            "agriculturalLandArea":                       "SE025",       # Total Area of type Agricultural Land [ha]
            "agriculturalLandValue":                      "D_CV_3010_V", # Total value of Agricultural Land [€]
            "agriculturalLandHectaresAdquisition":        "agriculturalLandHectaresAdquisition", # Acquired Agricultural Land [ha]
            "landImprovements":                           "D_CV_3020_V", # Invesment in Land improvements [€]
            "forestLandArea":                             "SE075",       # Total Area of type Forest Land [ha]
            "forestLandValue":                            "D_CV_5010_V", # Total value of Forest Land [€]
            "farmBuildingsValue":                         "D_CV_3030_V", # Value of Buildings in the farm [€]
            "machineryAndEquipment":                      "D_CV_4010_V", # Value of Machinery and Equipment in the farm [€]
            "intangibleAssetsTradable":                   "D_CV_7010_V", # Value of intangible assets that are tradable [€]
            "intangibleAssetsNonTradable":                "D_CV_7020_V", # Value of intangible assets that are non-tradable [€]
            "otherNonCurrentAssets":                      "D_CV_8010_V", # Value of other non-current assets [€]
            "longAndMediumTermLoans":                     "SE490", # Total value of established long and medium term loans [€]
            "totalCurrentAssets":                         "SE465", # Total value of current assets [€]
            "farmNetIncome":                              "SE420", # Farm Net Income [€]
            "grossFarmIncome":                            "SE410", # Gross Farm Income [€]
            "subsidiesOnInvestments":                     "SE406", # Total value of subsidies on investments [€]
            "vatBalanceOnInvestments":                    "SE408", # Balance of Taxes on Investments [€]
            "totalOutputCropsAndCropProduction":          "SE135", # Total value of Agricultural Production [€]
            "totalOutputLivestockAndLivestockProduction": "SE206", # Total value of Livestock Production [€]
            "otherOutputs":                               "SE256", # Total value of other outputs [€]
            "totalIntermediateConsumption":               "SE275", # Total value of intermediate consumption [€]
            "taxes":                                      "SE390", # Value of Taxes (>0 received , <0 paid) [€]
            "vatBalanceExcludingInvestments":             "SE395", # Balance of VAT excluding investments [€]
            "fixedAssets":                                "SE441", # Total value of Fixed Assets [€]
            "depreciation":                               "SE360", # Yearly Depreciation [€]
            "totalExternalFactors":                       "SE365", # Total value of External Factors [€]
            #"machinery":                                  "D_CV_4010_V", # Total value of Machinery [€] # Duplicated 
            "rentPaid":                                   "SE375", # Rent paid for land and buildings and rental cahnges [€]
            "rentBalance":                                "rentBalance", # Balance (>0 received , <0 paid) of rent operations [€]
            "specificCropCosts":                          "SE284", # Specific Crop costs
            "plantationsValue":                           "SE285", # Seeds and plants € Relates to agricultural and horticultural crops. New plantations of permanent crops and woodlands are considered as investments.
            }
            
        self.HolderFarmYearData_dict = {
            "holderAge":    "C_UR_10_A", #["C_UR_10_G", "C_UR_20_G"], 
            "holderGender": "C_UR_10_G", #["C_UR_10_B", "C_UR_20_B"], 
            
            # Not available
            "holderSuccessors":    "holderSuccessors", 
            "holderSuccessorsAge": "holderSuccessorsAge", 
            "holderFamilyMembers": "holderFamilyMembers", 
            "yearNumber":          "YEAR" # Note
            }
        
        # Note: althoug year is a parameter that appears in other agent fields, it has been included here
        # because no csv modification is required

        # Required to build organic version of crops although not required in the agents
        self.Organic_variables_dict = {
            "organicFarming":        "A_CL_140_C", # Organic farming 
            "sectorsOrganicFarming": "A_CL_141_C", # Sectors in organic farming
        }


    def _record_organic_type_variables(self):
        """
        Record or generate organic-type variables. If available in microdata, get from it. Otherwise, generate random values

        Returns
        ----------
        organic_df: pd.DataFrame
            values for the variables required to make (conventional) -> (conventional, organic) transformation. 
        """
        
        organic_df = pd.DataFrame()

        for k, fadn_var in zip(self.Organic_variables_dict.keys(), self.Organic_variables_dict.values()):
            
            df2 = self.microdata[fadn_var].copy(deep=True) if fadn_var in self.microdata.columns else self._fix_missing_organic_variables(k)
            
            # Merge dataframes
            organic_df = pd.concat([organic_df, df2], axis=1)

        organic_df = organic_df.rename(columns=dict(zip(self.Organic_variables_dict.values(), self.Organic_variables_dict.keys())))

        organic_df["organicFarming"] = organic_df["organicFarming"].fillna(1)
        organic_df["sectorsOrganicFarming"] = organic_df["sectorsOrganicFarming"].fillna(0)

        return organic_df
        
    
    def _fix_missing_organic_variables(self, var):
        """
        Generate synthetic data for organic variables using experimental data from Andalusian use case.
        
        Parameters
        ----------
        var: str
            name of the variable to be generated
        Returns
        ----------
        synthetic_sample: list
            synthetic sample of the
        """
        
        experimental_probabilities = {
            'organicFarming': {
                1: 0.7794350703831453, 3: 0.12911345203691618, 2: 0.0881886827631211, 4: 0.0032627948168173766},
            'sectorsOrganicFarming': {
                0.0: 0.8749883471613685, 34.0: 0.04185699636431435, 42.0: 0.030949939405239116, 33.0: 0.024983686025915913, 39.0: 0.014076629066840683, 31.0: 0.007178148596998229, 36.0: 0.0036356856530250768, 37.0: 0.001678008762934651, 35.0: 0.0005593362543115503, 38.0: 9.322270905192505e-05}}
        
        # Return values as dataframe
        synth_sample = pd.DataFrame({var: np.random.choice(list(experimental_probabilities[var].keys()), self.n_rows, experimental_probabilities[var].values)})

        return synth_sample
    

    def _get_totals_df(self):
        """
        
        """

        if self.USE_CASE in ["poland", "greece"]:
            weight_computer = WeightComputer(self.METADATA_PATH, self.USE_CASE, self.YEAR)
        
        # Add weight related variables to microdata if not available
        for var in ["A_TY_90_TF", "A_TY_90_ES", "A_LO_40_N", "A_LO_40_N2", "A_TY_80_W", ]:
            if not var in self.microdata.columns:
                self.microdata = weight_computer.main(self.microdata, var)
            
        # Create a dataframe for totals
        totals_variables_df = self.microdata[["A_TY_90_TF", "A_TY_90_ES", "A_LO_40_N", "A_LO_40_N2", "A_TY_80_W", ]].copy(deep=True)

        # Variables used to compute crop representativeness
        for var in ["B_UO_10_A", "B_UT_20_A", "B_US_30_A", "SE025", "SE135"]:
            if var in self.microdata.columns:
                totals_variables_df = pd.concat([totals_variables_df, self.microdata[var]], axis=1)

        return totals_variables_df
        

    def _transform_table_codes(self, data):
        """
        Adapt codes tables to RICA example.
        """


        df = data.copy(deep=True)
        
        df = df.rename(columns={
            "Description": "Description", 
            "code": "FADN Included products IDs", 
            "aggregation": "CUSTOM GROUP (EN)",
            "aggregation description": "species"})
        
        final_columns = ["species", "Description", "FADN Included products", "FADN Included products IDs", "CUSTOM GROUP (EN)", "arable", "aggregation_arable"]

        for c in final_columns:
            if not c in df.columns:
                df[c] = ""
        
        
        structure_chain = lambda i, x, len_: f"{x}, " if i<len_ else f"x"

        for i in df.index:
            if df.at[i, "FADN Included products IDs"] == df.at[i, "CUSTOM GROUP (EN)"]:
                df.at[i, "FADN Included products"] = df.at[i, "Description"]
            else:
                #
                all_products = df[df["CUSTOM GROUP (EN)"]==df.at[i, "CUSTOM GROUP (EN)"]]["Description"]
                included_products = "".join([structure_chain(i, prod, len(all_products)) for i, prod in enumerate(all_products)])
                df.at[i, "FADN Included products"] = included_products

        df = df[final_columns]
        
        return df
    

    def _transform_table_codes_policies(self, data):
        """
        Transform policy codes into AGRICORE format

        Parameters
        ----------
        data: pd.DataFrame
            data containing policy codes in the original format
        Returns
        ----------
        df: pd.DataFrame
            policy codes in proper format
        """

        df = data.copy(deep=True)
        #-----------------------------------------
        
        df = df.rename(columns={
            "Description": "Description", 
            "Code": "FADN Included codes IDs", 
            "aggregation description": "species"})
        

        final_columns = ["Description", "FADN Included codes IDs", "Coupled", "CUSTOM GROUP"]

        for c in final_columns:
            if not c in df.columns:
                df[c] = ""
        
        # Order columns
        df = df[final_columns]

        return df

    
    def _create_agricore_names(self):
        """
        Compose the list of colall variaboless required for the analysed use case.

        Parameters
        ----------

        Returns
        ----------
        spg_columns: list
            list of columns to be generated for the synthetic population
        """
        
        # Missing variables on microdata
        missing_variables = []
        spg_columns_agricore = []

        # Basic columns
        
        spg_columns_fadn = list(self.Farm_dict.values()) + list(self.ClosingValue_dict.values()) + list(self.HolderFarmYearData_dict.values())

        # Expand basic variables
        for table in [self.Farm_dict, self.ClosingValue_dict, self.HolderFarmYearData_dict]:
            for agri_name in table.keys():

                # Get FADN name
                fadn_name = table[agri_name]
                spg_columns_agricore.append(agri_name)

                # Create an empty column if it is not available on microdata
                if not fadn_name in self.microdata.columns:
                    self.microdata = pd.concat([self.microdata, pd.DataFrame({agri_name: [0.0]*self.n_rows})], axis=1)
                    missing_variables.append(fadn_name)
                else:
                    self.microdata = self.microdata.rename(columns={fadn_name: agri_name})

        # Expand the crop variables given the crop codes: CONVENTIONAL
        for group in self.crop_codes["CUSTOM GROUP (EN)"].dropna().unique():
            
            # Get the list of FADN codes composing the custom group
            unique_codes = list(self.crop_codes[self.crop_codes["CUSTOM GROUP (EN)"]==group]["FADN Included products IDs"].dropna().unique())
            
            # Identify organic type
            organic = True if group.startswith("ORG") else False
            
            # Iterate vars and codes
            for var in self.table_I.index:
                for code in unique_codes:

                    # Compose agricore and FADN names and add them to respective lists
                    agri_name = f"{code}.{var}" if not organic else f"ORG_{code}.{var}"
                    fadn_name = f'{self.table_I.at[var, "start"]}{code}{self.table_I.at[var, "end"]}' if not organic else f'ORG_{self.table_I.at[var, "start"]}{code}{self.table_I.at[var, "end"]}'
                    spg_columns_agricore.append(agri_name)
                    
                    # Create empty column if it is not available on microdata
                    if not fadn_name in self.microdata.columns:
                        value = 0.0 if not fadn_name.endswith("organicProductionType") else "conventional" if not organic else "organic"
                        self.microdata = pd.concat([self.microdata, pd.DataFrame({agri_name: [value for _ in range(self.n_rows)]})], axis=1)
                        missing_variables.append(fadn_name)
                    else:
                        self.microdata = self.microdata.rename(columns={fadn_name: agri_name})
        

        # Expand the animal variables given the animal and animal products codes
        for table in [self.table_J, self.table_K_milk, self.table_K_wool, self.table_K_eggs, self.table_K_manure, self.table_K_other]:
            for var in table.index:
                for code in self.animal_codes["FADN Included products IDs"]:
                    
                    # Compose agricore and FADN names and add them to respective lists
                    agri_name = f"{code}.{var}"
                    fadn_name = f'{table.at[var, "start"]}{code}{table.at[var, "end"]}'
                    spg_columns_agricore.append(agri_name)

                    # Create empty column if it is not available on microdata
                    if not fadn_name in self.microdata.columns:
                        self.microdata = pd.concat([self.microdata, pd.DataFrame({agri_name: [0.0]*self.n_rows})], axis=1)
                        missing_variables.append(fadn_name)
                    else:
                        self.microdata = self.microdata.rename(columns={fadn_name: agri_name})
        
        # FiX dairyCows variable
        self.microdata["261.dairyCows"] = self.microdata["261.numberOfAnimals"]

        # Expand the subsidy variables given the subsidy codes
        for var in self.table_M.index:
            for code in self.subsidies_codes["Subsidy_Code"].unique():
                # Compose agricore and FADN names and add them to respective lists
                agri_name = f"{code}.{var}"
                fadn_name = f'{self.table_M.at[var, "start"]}{code}{self.table_M.at[var, "end"]}'
                spg_columns_agricore.append(agri_name)

                # Create empty column if it is not available on microdata
                if not fadn_name in self.microdata.columns:
                    self.microdata = pd.concat([self.microdata, pd.DataFrame({agri_name: [0.0]*self.n_rows})], axis=1)
                    missing_variables.append(fadn_name)
                else:
                    self.microdata = self.microdata.rename(columns={fadn_name: agri_name})
        
        # Select final variables
        self.microdata = self.microdata[spg_columns_agricore]
        
        # Save missing_variables
        pd.DataFrame({"missing variables": missing_variables}).to_csv(os.path.join(self.RESULTS_PATH, f"missing_variables_{self.USE_CASE}_{self.YEAR}.csv"), index=False)


    def _fill_no_FADN_variables(self):
        """
        Generate values for variables that are not available in the FADN. 
        Values are assigned to "FADN" names as are the ones that appear on microdata dataframe. 
        """
        
        # Manage organic variables
        self._organic_crops_management()

    
    def _make_aggregations(self):
        """
        Method to make the required aggreagations for the different products and variables (crops, livestock, animal products
        and subsidies) according to each use case.

        Parameters
        ----------

        Returns
        ----------
        """
        
        # Aggregate crops
        self._aggregate(self.crop_codes, self.table_I)
        
        # Aggregate animals
        self._aggregate(self.animal_codes, self.table_J)

        # Aggregate animal products
        self._aggregate(self.animal_codes, self.table_K_milk)
        self._aggregate(self.animal_codes, self.table_K_manure)
        self._aggregate(self.animal_codes, self.table_K_eggs)
        self._aggregate(self.animal_codes, self.table_K_wool)
        self._aggregate(self.animal_codes, self.table_K_other)
        
        # Subsidies do not need to be aggregated
    

    def _aggregate(self, aggregation_codes, variables_table):
        """
        Make specific aggregations for a given product or set of variables.
        Aggreagate variables of the same type for the given codes.
        Update microdata file based on those aggregations. 

        Parameters
        ----------
        aggregation_codes: pd.DataFrame
            dataframe with the relations between original codes and aggregations to be performed
        variables_table: pd.DataFrame
            dataframe with the links between FADN variables and AGRICORE variables. In this case,
            this dataframe is used to evaluate the AGRICORE variables.
        Returns
        ----------
        """

        for agg  in aggregation_codes["CUSTOM GROUP (EN)"].dropna().unique():
            
            organic = "ORG_" if agg.startswith("ORG") else ""

            # Get the codes that should be aggregated
            codes_agg = aggregation_codes[aggregation_codes["CUSTOM GROUP (EN)"]==agg]["FADN Included products IDs"].to_list()
            
            # If aggregation is necessary, make the aggregation
            if len(codes_agg) > 0:
                
                # Inspect individual variables
                for var in variables_table.index:
                    
                    # Get list of old variables from microdata
                    variables_to_aggregate = [f"{organic}{code}.{var}" for code in codes_agg if f"{organic}{code}.{var}" in self.microdata.columns]

                    if not var in self.categorical_variables:
                        
                        # Sum of the variables included in the above list
                        self.microdata[f"{agg}.{var}"] = self.microdata[variables_to_aggregate].apply(lambda x: x.sum(), axis=1)
                        
                        # Remove the variables used to make the aggregations if necessary        
                        self.microdata = self.microdata.drop(columns=variables_to_aggregate)
                        
                    else:
                        # Select the organic production type. If only one organic production type, keep this value. Otherwise assign undetermined
                        self.microdata[f"{agg}.{var}"] = self.microdata[variables_to_aggregate].apply(lambda x: x.unique()[0] if x.nunique()==1 else "undetermined", axis=1)

                        # Remove the variables used to make the aggregations if necessary        
                        self.microdata = self.microdata.drop(columns=variables_to_aggregate)
                    
            else:
                pass

    
    def _organic_crops_management(self):
        """
        Method to assign to conventional and organic production for a crop code their values given a set of conditions

        Parameters
        ----------
        
        Returns
        ----------
        
        """
        
        # Get organic variables
        organicFarming_values = self.organic_variables["organicFarming"].values
        sectorsOrganicFarming_values = self.organic_variables["sectorsOrganicFarming"].values
        
        # Apply logic
        # Organic farming
        # 1 = the holding does not apply organic production methods
        # 2 = the holding applies only organic production methods for all its products
        # 3 = the holding applies both organic and other production methods
        # 4 = the holding is converting to organic production methods
        
        # Sectors in organic farming
        # * To be filled only if the holding applies both organic and other production  methods.
        # 0 = not applicable (the holding applies both organic and other production methods  for all its sectors of production)
        # * Codes indicating the sectors of production where the holding applies only  organic production method (multiple selections are allowed):
        # 31 = cereals 
        # 32 = oilseeds and protein crops 
        # 33 = fruits and vegetables (including citrus fruits, but excluding olives) 
        # 34 = olives 
        # 35 = vineyards 
        # 36 = beef 
        # 37 = cow’s milk 
        # 38 = pigmeat 
        # 39 = sheep and goats (milk and meat) 
        # 40 = poultry meat 
        # 41 = eggs 
        # 42 = other sector

        # 0 = not applicable (the holding applies both organic and other production methods  for all its sectors of production)
        # 31 = cereals 
        # 32 = oilseeds and protein crops 
        # 33 = fruits and vegetables (including citrus fruits, but excluding olives) 
        # 34 = olives 
        # 35 = vineyards 
        
        IDs_for_organic_sector= {
            #0: self.crop_codes["FADN Included products IDs"].unique().tolist(), # All
            0: self.crop_codes[self.crop_codes["CUSTOM GROUP (EN)"].isin(self.crop_codes["CUSTOM GROUP (EN)"].dropna().unique())]["FADN Included products IDs"].unique().tolist(), 
            31: [10110, 10120, 10130, 10140, 10150, 10160, 10170, 10190], 
            32: [10210, 10220, 10290 ], 
            33: [10300, 10310, 10390, 10711, 10712, 10720, 10731, 10732, 10733, 10734, 10735, 10736, 10737, 10738, 10739, 10790], 
            34: [40310, 40320, 40330, 40340], 
            35: [40411, 40412, 40420, 40430, 40440, 40451, 40452, 40460, 40470], 
            }

        conv_custom_group = [agg for agg in self.crop_codes["CUSTOM GROUP (EN)"].dropna().unique() if not agg.startswith("ORG")]
        conv_individual_codes = self.crop_codes[self.crop_codes["CUSTOM GROUP (EN)"].isin(conv_custom_group)]["FADN Included products IDs"].dropna().unique().tolist()
        org_custom_group = [agg for agg in self.crop_codes["CUSTOM GROUP (EN)"].dropna().unique() if agg.startswith("ORG")]
        org_individual_codes = self.crop_codes[self.crop_codes["CUSTOM GROUP (EN)"].isin(org_custom_group)]["FADN Included products IDs"].dropna().unique().tolist()
        
        # Fix unavailable organic sector
        for os in sorted(pd.DataFrame(sectorsOrganicFarming_values)[0].unique()):
            if os not in IDs_for_organic_sector.keys():
                IDs_for_organic_sector[os] = self.crop_codes["FADN Included products IDs"].unique().tolist(), # All

        # If of in  [1, ] -
        for i, of in enumerate(organicFarming_values):
            for crop in conv_individual_codes:
                # TODO: Move this outside the loop for increased performance
                # TODO: Consider properly the Undeterined possibility only if a product group for that fadn code exist and has the Organic Field set as Undetermined.
            
                # 1. Check the crop in crops_codes
                crop_info = self.crop_codes[self.crop_codes["FADN Included products IDs"]==crop][["CUSTOM GROUP (EN)"]].copy(deep=True).drop_duplicates()

                # 2. Check if organic is defined for the aggregation
                crop_info["organic available"] = crop_info.apply(lambda x: 1 if x["CUSTOM GROUP (EN)"].startswith("ORG_") else 0, axis=1)

                # 3.1 If ORG is defined, proceed with the above code
                if crop_info["organic available"].sum() > 0:
                    
                    for var in self.table_I.index:
                            agricore_name = f"{crop}.{var}"
                            
                            if var != "organicProductionType":
                                # Only conventional production methods 
                                if of==1:
                                    pass
                                # Only organic production methods 
                                elif of==2:
                                    # Move all numerical variables values to organic
                                    # -> assign to organic columns the values of conventional and fill conventional with zeros
                                    self.microdata.at[i, f"ORG_{agricore_name}"] = self.microdata.at[i, f"{agricore_name}"]
                                    self.microdata.at[i, f"{agricore_name}"] = 0
                                elif of==3 or of==4:
                                    if crop in IDs_for_organic_sector[sectorsOrganicFarming_values[i]]:
                                        # As the ration between conventional and organic is unknown since data comes merged for both production types, no numerical values
                                        # modification will be applied
                                        organic_ratio = 0.25
                                        # Apply organic ratio to the organic variable
                                        self.microdata.at[i, f"ORG_{agricore_name}"] = self.microdata.at[i, f"{agricore_name}"]*organic_ratio
                                        # Reduce conventional value according to organic ratio
                                        self.microdata.at[i, f"{agricore_name}"] = self.microdata.at[i, f"{agricore_name}"]*(1 - organic_ratio)
                                    else:
                                        pass
                            # else:
                            #     # Change organicProductionType if organicFarming is 3 or 4 and the crop in question is within the organic sectors where the farm applies organic production methods
                            #     self.microdata.at[i, f"{agricore_name}"] = "Undetermined" if (of==3 or of==4) and crop in IDs_for_organic_sector[sectorsOrganicFarming_values[i]] else "Conventional"

                # Organic is not defined for the current crop. No organic operations are requiered
                else:
                    pass

        
    def _make_corrections(self):
        """
        Make corrections in the microdata. Add variables that was not possible to add during the microdata processing.
        """

        # Correction 1: add machinery column
        # This variables was not possible to be added since another variable machineryAndEquipment has the same FADN code. As result of this,
        # problems arisen when making microdata column renaming
        self.microdata["machinery"] = self.microdata["machineryAndEquipment"].apply(lambda x: x)

        # Correction 2: replace Nans by zeros
        self.microdata = self.microdata.fillna(0)

        # Correction 3: convert holder age to int
        self.microdata["holderAge"] = self.microdata["holderAge"].apply(lambda x: int(x))
        
        # Correction 4: landValue assignation
        self._land_value_inputer()

        # Correction 5: rentPaid[€] -> rentPaid[€/Ha]
        self.microdata["rentPaid"] = self.microdata.apply(lambda x: x["rentPaid"]/x["rentBalanceIn"] if x["rentBalanceIn"]>0 else 0.0, axis=1)
        

    def _land_value_inputer(self):
    
        # Load landValue data
        landValue = pd.read_csv(f"{self.METADATA_PATH}/land_value.csv")

        # Include product groups in landValue
        productMapping = self.crop_codes[self.crop_codes["CUSTOM GROUP (EN)"].apply(lambda x: not x.startswith("ORG"))]
        productMapping = dict(zip(productMapping["FADN Included products IDs"].tolist(), productMapping["CUSTOM GROUP (EN)"].tolist()))
        landValue["CUSTOM GROUP (EN)"] = landValue.apply(lambda x: productMapping[x["Crop"]], axis=1)

        # Compute land value for each product group
        for agg in self.crop_codes["CUSTOM GROUP (EN)"].unique():
            FADN_CROPS = self.crop_codes[self.crop_codes["CUSTOM GROUP (EN)"] == agg]["FADN Included products IDs"].unique()
            
            for crop in FADN_CROPS:
                irrigated_price = landValue[landValue["Crop"] == crop]["Value irr"].item()
                dry_price       = landValue[landValue["Crop"] == crop]["Value dry"].item()
                dry_price = dry_price if dry_price > 0 else irrigated_price

                # Compute weighted landValue
                # [€] = [€/Ha]·[Ha ]
                self.disagregated_crops[f"{crop}.landValue"] = self.disagregated_crops.apply(lambda x: x[f"{crop}.irrigatedArea"]*irrigated_price + (x[f"{crop}.cultivatedArea"] - x[f"{crop}.irrigatedArea"])*dry_price, axis=1)
            
            # Compute aggregated landValue
            variables_to_aggregate = [f"{crop}.landValue" for crop in FADN_CROPS]

            self.microdata[f"{agg}.landValue"] = self.disagregated_crops[variables_to_aggregate].sum(axis=1)


    def _fill_categorical(self):
        """
        Auxiliar method to fill the categorical columns with normal values

        Parameters
        ----------

        Returns
        ----------
        """
                        
        self.n_rows = self.n_rows

        for cat_var in self.categorical_variables:
            for agr_var in self.microdata.columns:
                if agr_var.endswith(cat_var):
                    
                    if (self.microdata[agr_var].nunique()==1 and (self.microdata[agr_var].unique()==[0.0] or self.microdata[agr_var].unique()==[0])):

                        if cat_var=="technicalEconomicOrientation":
                            self.microdata[agr_var] = self._generate_categorical_data([1510, 1520, 1660, 2110, 3510, 3610, 3630, 4500, 4600, 4820, 4820, 5120, 5220, 8330, 8340, ], self.n_rows)
                        elif cat_var=="altitude":
                            self.microdata[agr_var] = self._generate_categorical_data([1, 2, 3], self.n_rows)
                        elif cat_var=="weight_ra":
                            example_weight = {4: 386, 61: 269, 62: 169, 240: 163, 241: 191, 64: 339, 71: 180, 73: 143, 74: 214, 288: 257, 306: 128, 16: 246, 274: 127, 313: 125, 17: 270, 68: 134, 26: 612, 28: 229, 19: 260, 18: 274, 31: 413, 79: 103, 45: 208, 83: 192, 5: 261, 29: 89, 90: 126, 22: 133, 24: 146, 97: 86, 14: 142, 32: 122, 98: 97, 33: 84, 35: 200, 105: 74, 37: 164, 57: 105, 60: 156, 39: 161, 69: 68, 55: 93, 56: 128, 47: 127, 51: 78, 59: 152, 65: 81, 20: 73, 120: 74, 38: 66, 58: 65, 127: 54, 30: 71, 23: 110, 15: 66, 92: 45, 160: 43, 6: 99, 219: 42, 25: 73, 34: 78, 106: 39, 181: 38, 41: 74, 191: 36, 43: 35, 52: 34, 44: 47, 126: 33, 42: 39, 76: 31, 146: 36, 12: 30, 13: 44, 72: 28, 103: 27, 162: 27, 139: 26, 36: 41, 75: 21, 154: 21, 263: 21, 107: 20, 77: 19, 242: 19, 49: 22, 1: 18, 262: 24, 116: 18, 255: 18, 53: 17, 270: 17, 54: 16, 10: 29, 7: 15, 179: 14, 209: 12, 87: 11, 439: 11, 239: 19, 251: 10, 93: 10, 8096: 10, 21: 22, 343: 9, 279: 9, 91: 9, 110: 8, 63: 8, 197: 8, 9: 12, 314: 8, 4382: 7, 225: 7, 206: 7, 441: 7, 361: 6, 66: 5, 164: 5, 310: 4, 1325: 4, 232: 4, 389: 4, 497: 4, 1804: 4, 86: 3, 399: 3, 214: 3, 413: 3, 11: 4, 1665: 2, 666: 2, 185: 2, 40481: 1, 229: 1, 132: 1}
                            self.microdata[agr_var] = random.choices(list(example_weight.keys()), weights=list(example_weight.values()), k=self.n_rows)
                        elif cat_var=="holderAge":
                            self.microdata[agr_var] = pd.DataFrame(np.random.normal(50, 5, self.n_rows))
                        elif cat_var=="holderGender":
                            self.microdata[agr_var] = self._generate_categorical_data(
                                ["Male", "Female"], self.n_rows)
                        elif cat_var=="holderSuccessors":
                            self.microdata[agr_var] = [int(v) for v in np.random.lognormal(.5, .5, self.n_rows)]
                        elif cat_var=="holderSuccessorsAge":
                            self.microdata[agr_var] = pd.DataFrame(np.random.normal(30, 5, self.n_rows))
                        elif cat_var=="holderFamilyMembers":
                            self.microdata[agr_var] = pd.DataFrame([int(v) for v in np.random.normal(3, 0.75, self.n_rows)])[0].apply(lambda x: x if x!=0 else 1).tolist()
                        else:
                            pass
        
    
    def _generate_categorical_data(self, categories, size):
        """
        Generate random categorical data given a list of categories

        Paramters
        ---------
        categories: list
            list containing the categories that must contain the generated data
        return: list
            list containing the random sample
        """
        cat_number = {}

        for i, cat in enumerate(categories):
            cat_number[i] = cat
        
        gen = np.random.randint(min(list(cat_number.keys())), max(list(cat_number.keys()))+1, size)

        result = pd.DataFrame({"gen": gen})

        result["gen"] = result["gen"].apply(lambda x: cat_number[x])

        return result


    def _generate_totals(self): 
        """
        Method used to generate totals from the original dataset
        
        For each prouct, totals is computed as follows:
        totals_population_{product_i} = totals_sample(OTE, ES, NUTS3)_{product_i} * weights

        where:
            totals_population: totals of the population
            totals_sample: totals of the sample. Sum for all the instances belonging to the group composed by OTE, ES and NUTS3 for each product
            product_i: product used to compute totals
            OTE: techno-ecnomic orientation of the farms composing the group
            ES: economic size of the farms composing the group
            NUTS3: region to which the farms composing the group belong
            weights: (n_farms_population/n_farms_sample), extrapolation ratio of the given sample group
            
        Parameters
        ----------

        Returns
        ----------
        totals: pd.Dataframe
            dataframe with the value of total values for:
            * total area for all the crops available
            * total production for all the crops available
            * total production for all the livestock available

        number_of_farms_population: int
            number of farms composing the population inferred from the sample and the weights
        """
        microdata = self.microdata.copy(deep=True)

        # Merge microdata and totals_variables_df
        microdata = pd.concat([microdata, self.totals_variables_df], axis=1)
        
        # Declare some relevant variables for the task
        var_weights = "A_TY_80_W"
        var_OTE = "A_TY_90_TF"
        var_ES = "A_TY_90_ES"
        var_NUTS2 = "A_LO_40_N2"
        var_NUTS3 = "A_LO_40_N"
        
        totals_df = pd.DataFrame()

        for var in self.totals_variables:
            sameType_variables = [c for c in microdata.columns if var in c]
            for var_crop in sameType_variables:
                totals_df[f"{var_crop}"] = microdata.apply(lambda x: x[var_crop]*x[var_weights], axis=1)
            totals_df[f"TOTAL-{var}"] = totals_df.apply(lambda x: x[sameType_variables].sum(), axis=1)

        totals_df = pd.concat([totals_df, microdata[[var_weights, var_OTE, var_ES, var_NUTS2, var_NUTS3]]], axis=1)

        # Compute farms represented
        totals_df["farmsRepresented"] = totals_df.apply(lambda x: int(x[var_weights]), axis=1)

        # display(totals_df)
        # Save results
        totals_df.to_csv(self.TOTALS_PATH, index=False)
        
        print(f"Saved file  on {self.TOTALS_PATH}")
        

    def _check_totals_variables_df(self):
        """
        Check if the totals variables dataframe is properly filled. Otherwise, fill it using microdata values

        Economic size was alredy generated when totals_variables_df was generated
        
        """
        
        for fadn_var, agr_var in zip(["A_TY_80_W", "A_TY_90_TF", "A_LO_40_N"], ["weight_ra", "technicalEconomicOrientation", "regionLevel2"]):
            # If this condition is true, the column should be filled from microdata
            if (self.totals_variables_df[fadn_var].nunique()==1 and self.totals_variables_df[fadn_var].unique()[0]==0.0):
                self.totals_variables_df[fadn_var] = self.microdata[agr_var].apply(lambda x: x)

                
    def _standarsise_columns_units(self):
        """
        Standarsise columns units for all the columns in the dataset.
        Available FADN units:
            * Financial values: in euro or national currency except for Hungary which reports in 
            thousands of national currency units.
            * Physical quantities: in quintals (1 q = 100 kg), except in the case of eggs, which 
            will be expressed in thousands and wine and related products, which will be expressed 
            in hectolitres.
            * Areas: in ares (1 a = 100 m2, 1 hectare = 100 ares), except mushrooms to be expressed 
            in square metres of total cropped area and except in Table M “Subsidies”, where basic 
            units (in case they represent area) are to be registered in hectares (ha).
            * Average livestock numbers: to two decimal places.
            * Labour units: to two decimal places.
            * Basic units of subsidies (in table M): to two decimal places.
            * Codes: integers as listed in respective descriptions to the tables.
            * Other data: (e.g. geo-coordinates) as explained in respective descriptions to the tables.

        Synthetic population final units
            * Financial values          [€]
            * Physical quantities       [tons]
            * Physical quantities eggs  [100]
            * Areas                     [ha]
            * Average livestock numbers [units]

        Required converions
            * Financial values              [€] -> [€]     Conversion Factor  CF_fv = 1 -> not required
            * Physical quantities           [q] -> [tons]  Conversion Factor  CF_pq = 0.1
            * Physical quantities eggs    [100] -> [units] Conversion Factor CF_pqe = 1e2
            * Areas                         [a] -> [ha]    Conversion Factor  CF_a  = 1e-2
            * Average livestock numbers [units] -> [units] Conversion Factor  CF_u  = 1 -> not required
        """
        # Generic function to select all columns in microdata from a specific type
        _columns_selection = lambda columns_set: [c for c in self.microdata.columns for var in columns_set if c.endswith(var)]

        CF_fv  = 1      # Financial values
        CF_pq  = 1e-1 if self.USE_CASE=="andalusia" else 1  # Physical quantities
        CF_pqe = 1e2    # Physucal quantities eggs
        CF_a   = 1e-2 if self.USE_CASE=="andalusia" else 1   # Areas
        CF_aln = 1      # Average livestock numbers
        CF_sp  = 1    if self.USE_CASE=="andalusia" else 1   # Prices

        conversion_dict = {
            "financial values": {
                "conversion factor": CF_fv,
                "variables": ["valueSales", "cropProduction", "quantityUsed", "valueSoldAnimals", "valueAnimalsRearingBreading", "valueSlaughteredAnimals", "milkTotalSales", "eggsTotalSales",],},
            "physical quantities": {
                "conversion factor": CF_pq,
                "variables": ["milkTotalProduction", "milkProductionSold", "woolTotalProduction", "woolProductionSold", "quantitySold"],},
            "physical quantities eggs": {
                "conversion factor": CF_pqe,
                "variables": ["eggsTotalProduction", "eggsProductionSold"],}, 
            "areas": {
                "conversion factor": CF_a,
                "variables": ["irrigatedArea", "cultivatedArea", "rentBalanceIn", "rentBalanceOut"],},
            "units": {
                "conversion factor": CF_aln,
                "variables": ["numberOfAnimals", "numberOfAnimalsSold", "numberAnimalsRearingBreading", "numberAnimalsForSlaughtering", "dairyCows", ], 
            "prices":{
                "conversion factor": CF_sp,
                "variables": ["sellingPrice", ],
            }},
            #"financial vs physical quantities": {
            #    "conversion factor": CF_fv/CF_pq,
            #    "variables": ["irrigatedArea", "cultivatedArea"],},
            
        }

        # Make the conversion of all variables types
        for type_ in conversion_dict.keys():
            
            # Selct conversion factor
            cf = conversion_dict[type_]["conversion factor"]

            # Select the columns to be converted
            columns_selection = _columns_selection(conversion_dict[type_]["variables"])

            # Make the convrsion for the selected columns with the conversion factor
            self.microdata[columns_selection] = self.microdata[columns_selection].apply(lambda x: x*cf, axis=1)


    def _get_categoricals(self):
        """

        Get a list with all the categorical variables present in the microdata

        Parameters
        ----------
        
        Returns
        ----------
        categoricals_final_list: list
            final list of categorical variables
        """

        categoricals_final_list = []

        # Expand categorical variables
        for cat in self.categorical_variables:
            
            if cat in self.table_I.index:
                for code in self.crop_codes["CUSTOM GROUP (EN)"].dropna().unique():
                    categoricals_final_list.append(f"{code}.{cat}")
                    
            elif cat in self.animal_variables.index:
                for code in self.animal_codes["CUSTOM GROUP (EN)"].unique():
                    categoricals_final_list.append(f"{code}.{cat}")

            elif cat in self.table_M.index:
                for code in self.subsidies_codes["Subsidy_Code"].unique():
                    categoricals_final_list.append(f"{code}.{cat}")
                    
            else:
                categoricals_final_list.append(cat)

        return categoricals_final_list
    

    def _generate_numerical(self):
        """
        Assign values for specific variables that do not lye on random assignments.
        """

        # sellingPrice
        for agg in self.crop_codes["CUSTOM GROUP (EN)"].dropna().unique():
            
            # Perform basic computing
            self.microdata[f"{agg}.sellingPrice"] = self.microdata.apply(lambda x: x[f"{agg}.valueSales"] / x[f"{agg}.quantitySold"] if x[f"{agg}.quantitySold"] > 0 else 0.0, axis=1)

            # Select variables with price higher than zero
            avg_price = self.microdata[self.microdata[f"{agg}.sellingPrice"]>0][f"{agg}.sellingPrice"].mean()
            
            if np.isnan(avg_price):
                avg_price = 0.0

            indexes_sellingPrice = self.microdata[(self.microdata[f"{agg}.sellingPrice"]==0)&(self.microdata[f"{agg}.valueSales"]>0)].index

            # Input mean value
            self.microdata.loc[indexes_sellingPrice, f"{agg}.sellingPrice"] = np.ones(len(indexes_sellingPrice))*avg_price

            #self.microdata[f"{agg}.sellingPrice"] = self.microdata.apply(lambda x: x[f"{agg}.sellingPrice"] if x[f"{agg}.sellingPrice"] > 0 else avg_price, axis=1)

        self.microdata = self.microdata.fillna(0)
        
        # cropProduction: unit change
        # Quantity -> Value
        # cropProduction[€] = cropProudction[Ton]*sellingPrice[€/Ton]
        for agg in self.crop_codes["CUSTOM GROUP (EN)"].dropna().unique():            
            self.microdata[f"{agg}.cropProduction"] = self.microdata.apply(lambda x: x[f"{agg}.cropProduction"] * x[f"{agg}.sellingPrice"], axis=1)
        
        # quantityUsed: unit change
        # Value -> quantity
        # quantityUsed[Ton] = quantityUsed[€]/sellingPrice[€/Ton]
        for agg in self.crop_codes["CUSTOM GROUP (EN)"].dropna().unique():            
            self.microdata[f"{agg}.quantityUsed"] = self.microdata.apply(lambda x: x[f"{agg}.quantityUsed"] / x[f"{agg}.sellingPrice"] if x[f"{agg}.sellingPrice"] > 0 else 0.0, axis=1)
        

    def main(self):
        """
        Update metadata object to create links between the different nomenclatures
        """
        
        # Create AGRICORE names from FADN
        self._create_agricore_names()
        
        # Build some population variables that can not be directly obtained from microdata
        self._fill_no_FADN_variables()
        
        # Compute crop representativeness
        rep_class = RepresentativenessComputer(self.microdata, self.crop_codes, self.totals_variables_df.copy(deep=True))
        representativeness = rep_class.compute_indicators()
        representativeness.to_csv(os.path.join(self.RESULTS_PATH, f"FADN_Representativeness_{self.YEAR}.csv"), index=False)
        
        # Standarsise columns units
        self._standarsise_columns_units()

        self.disagregated_crops = self.microdata.copy(deep=True)
        
        # Make aggregations
        self._make_aggregations()
        
        # Make corrections
        self._make_corrections()
        
        # Fill and adjust categorical variables
        #self._fill_categorical()

        # Get the final list of categorical variables
        categoricals_final_list = self._get_categoricals()
        
        # Generate numerical variables from microdata
        self._generate_numerical()
        
        # Generate totals file
        #self._generate_totals()
        
        return self.microdata, categoricals_final_list, self.totals_variables_df["A_TY_80_W"]
    