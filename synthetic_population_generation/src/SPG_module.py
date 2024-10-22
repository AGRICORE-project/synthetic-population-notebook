from uu import encode
import pandas as pd
import numpy as np 
import random
import math
import os
from scipy.stats import ttest_ind
from itertools import product
import time

from dask import config as cfg
from datetime import datetime

from src.generation_functions import _generate_synthetic_sample, _knn, _get_mean_std_var, _standarization_function, _KDE_function, _generate_column, _decode_data

from src.VariableNameConversion import VariableNameConversion
from complete_execution_andalucia.src.DAG_generator import DAG_generator
from src.italian_name_conversion import italian_name_conversion
from src.region_level_filler import RegionLevelSpain, RegionLevelGreece, RegionLevelPoland

from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

from sklearn.preprocessing import OrdinalEncoder


class SyntheticPopulationGenerator():
    """
    CLASS DESCRIPTION
    Class used to generate synthetic populations given a dataset in tabular format.
    """

    def __init__(self, number_of_farms_population, base_path, use_case, year, threads_per_worker=1, n_workers=1, totals_default=["cultivatedArea", "irrigatedArea", "cropProduction"]):

        """
        Parameters
        ----------
        use_case: str
            name of the country of the use case
        year: int
            year to be replicated in the synthetic population
        """
        self._KDE_list = []
        self._KNN_list = []
        self._OTHER = {}

        self.requested_number_of_farms = number_of_farms_population

        self.USE_CASE = use_case
        self.YEAR = year

        # PATHS ----------------------------------------------------------------------------------------------
        # Path where microdata and metadata is stored for hte given use case
        self.BASE_PATH = base_path
        METADATA_PATH = os.path.join(self.BASE_PATH, "metadata")
        self.RESULTS_PATH = os.path.join(self.BASE_PATH, "results")

        # Common to all use cases
        FORBIDDEN_DIRECTIONS_FILENAME = "forbidden_directions_v2.csv"
        CATEGORICAL_VARIABLES_FILENAME = "categorical_variables.csv"

        # Totals file has been generated artifically. It need to be fixed
        TOTALS_FILENAME = f'totals_{self.YEAR}.csv'
        N_FARMS_FILENAME = f"number_of_farms.csv"

        # Input files paths 
        CROPS_CODES_PATH = "Product_Mapping.csv"
        ANIMAL_CODES_PATH = "animal_codes.csv"
        ANIMAL_PRODUCTS_CODES_PATH = "animal_products_codes.csv"
        SUBSIDIES_CODES_PATH = "subsidies.csv"

        DISTRICTS_FILENAME = "n_holdings_district.csv"
        VARIABLE_COSTS_CROP_FILENAME = f"Crop-costs-deploy.csv"

        self.TOTALS_PATH = os.path.join(METADATA_PATH, TOTALS_FILENAME)
        self.NUMBER_OF_FARMS_PATH = os.path.join(METADATA_PATH, N_FARMS_FILENAME)
        FORBIDDEN_DIRECTIONS_PATH = os.path.join(METADATA_PATH, FORBIDDEN_DIRECTIONS_FILENAME)
        self.CATEGORICAL_PATH = os.path.join(METADATA_PATH, CATEGORICAL_VARIABLES_FILENAME)
        self.DISTRICT_FILEPATH = os.path.join(METADATA_PATH, DISTRICTS_FILENAME)
        self.VARIABLE_COSTS_CROP_FILEPATH = os.path.join(METADATA_PATH, VARIABLE_COSTS_CROP_FILENAME)

        SYNTHETIC_POPULATION_FOLDER = "synthetic_population"
        if not "synthetic_population" in os.listdir(self.BASE_PATH):
            os.mkdir(os.path.join(self.BASE_PATH, SYNTHETIC_POPULATION_FOLDER))

        self.SYNTHETIC_POPULATION_PATH = os.path.join(self.BASE_PATH, SYNTHETIC_POPULATION_FOLDER)
        # --------------------------------------------------------------------------------------------------------------

        # Load input files----------------------------------------------------------------------------------------------
        self.crop_codes = pd.read_csv(os.path.join(METADATA_PATH, CROPS_CODES_PATH))
        
        if False:
            # Remove SET_ASIDE group
            self.crop_codes = self.crop_codes[self.crop_codes["CUSTOM GROUP (EN)"]!="SET_ASIDE"]

        self.crop_codes = self.crop_codes["CUSTOM GROUP (EN)"].drop_duplicates().dropna().unique().tolist()
        
        # Compose unique animal codes
        self.animal_codes = pd.concat([
            pd.read_csv(os.path.join(METADATA_PATH, ANIMAL_PRODUCTS_CODES_PATH))["aggregation"],
            pd.read_csv(os.path.join(METADATA_PATH, ANIMAL_CODES_PATH))["aggregation"]
        ]).unique().tolist()

        self.subsidy_codes = pd.read_csv(os.path.join(METADATA_PATH, SUBSIDIES_CODES_PATH))["Subsidy_Code"].unique().tolist()
        
        # Load forbidden directions
        self.forbidden_directions_rules = pd.read_csv(FORBIDDEN_DIRECTIONS_PATH)
        
        # --------------------------------------------------------------------------------------------------------------
        
        self.group1 = ["lat", "long", "altitude", "farmCode", "technicalEconomicOrientation", "weight_ra", ]
        self.group2 = ["regionLevel1Name", "regionLevel2Name", "regionLevel3Name", "regionLevel1", "regionLevel2", "regionLevel3", "weight_reg", "rentBalanceIn", "rentBalanceOut"]
        self.group3 = ["agriculturalLandArea", "agriculturalLandValue", "agriculturalLandHectaresAdquisition", "landImprovements", "forestLandArea", "forestLandValue", "farmBuildingsValue", "machineryAndEquipment", "intangibleAssetsTradable", "intangibleAssetsNonTradable", "otherNonCurrentAssets", "specificCropCosts", "plantationsValue"]
        self.group4 = ["longAndMediumTermLoans", "totalCurrentAssets", "farmNetIncome", "grossFarmIncome", "subsidiesOnInvestments", "vatBalanceOnInvestments", "totalOutputCropsAndCropProduction", "totalOutputLivestockAndLivestockProduction", "otherOutputs", "totalIntermediateConsumption", "taxes", "vatBalanceExcludingInvestments", "fixedAssets", "depreciation", "totalExternalFactors", "machinery", "rentBalance", "rentPaid"]

        self.group5 = ["holderAge", "holderGender", "holderSuccessors", "holderSuccessorsAge", "holderFamilyMembers", "yearNumber", ]

        self.crops_variables = ["quantitySold", "valueSales", "cropProduction", "irrigatedArea", "cultivatedArea", "organicProductionType", "variableCostsCrops", "landValue", "quantityUsed", "sellingPrice"]

        self.animal_variables = ["numberOfAnimals", "numberOfAnimalsSold", "valueSoldAnimals", "numberAnimalsRearingBreading", "valueAnimalsRearingBreading", "numberAnimalsForSlaughtering", "valueSlaughteredAnimals", "milkTotalProduction", "milkProductionSold", "milkTotalSales", "milkVariableCosts", "dairyCows", "variableCostsAnimals", "woolTotalProduction", "woolProductionSold", "eggsTotalProduction", "eggsProductionSold", "eggsTotalSales", "manureTotalSales", ]
        self.subsidy_variables = ["value", ]
        
        # Set of variables with which totals are computed
        self.totals_variables = totals_default

        self.threads_per_worker = threads_per_worker
        self.n_workers = n_workers

    
    def _load_and_preprocess_microdata(self):
        """
        Load microdata and standarise nomenclature to AGRICORE names

        Preprocess microdata to select, clean and transform microdata features to get a dataframe ready to be ingested in the 
        Bayessian Network

        Parameters
        ----------
        microdata: pd.DataFrame
            microdata to be processed

        Returns
        ----------
        microdata: pd.DataFrame
            microdata processed
        
        Parameters
        ----------

        Returns
        ----------
        microdata: pd.DataFrame
            original population to be replicated using Bayessian Network
        """
        if self.USE_CASE != "italy":
            # Instantiate Variable name Conversor
            name_conversor = VariableNameConversion(self.BASE_PATH, self.USE_CASE, self.YEAR, self.totals_variables)
            
            # Apply name conversion. Get the list of categorical variables for the given use case
            microdata, self.categoricals, weights = name_conversor.main()

        else:
            microdata, self.categoricals, weights = italian_name_conversion(
                self.BASE_PATH, self.USE_CASE, self.YEAR, self.crops_variables, self.crop_codes, self.totals_variables)
            self.MICRODATA = microdata.copy(deep=True)
        

        # List of microdata variables
        self.columns_spg = list(microdata.columns)
        
        # Get the number of rows of y
        self.n_rows = microdata.shape[0]

        # Encode categoricals
        microdata = self._encode_categoricals(microdata)

        return microdata, weights


    def _encode_categoricals(self, microdata):
        """
        Encode categorical variables for their later conversion to numbers. Bayesian Network only is able
        to work with numerical data, so an encoding processing is required to generate the synthetic population.

        Parameters
        ----------
        microdata: pd.DataFrame
            data frame with all the use case variables. In this data frame, categorical variables appear in their
            original format
        
        Returns
        ----------
        y_encoded: pd.DataFrame
            data frame with encoded categorical variables
        """
        
        # Split microdata into numerical and categorical data
        microdata_num = microdata[[c for c in microdata.columns if c not in self.categoricals]].copy(deep=True)
        
        # Convert data type
        microdata_cat = microdata[self.categoricals].copy(deep=True).astype(str)
        
        # Instantiate data encoder
        self.encoder = OrdinalEncoder()

        # Encode categorical data
        encoded_data = self.encoder.fit_transform(microdata_cat)

        # Convert encoded data into data frame
        self.encoded_df = pd.DataFrame(encoded_data, columns=microdata_cat.columns)
        
        # Add encoded data to numerical data
        y_encoded = pd.concat([microdata_num, self.encoded_df], axis=1)

        return y_encoded
    
    
    def _build_bayesian_network(self, microdata, method="DAG"):
        """
        Method used to create the bayesian network using R
        Parameters
        ----------

        Returns
        ----------
        topological_order: list
            topologically sorted nodes of the bayesian network
        adj_mat_s: pd.DataFrame
            adjacency matrix representing the  bayesian network topologically sorted 
        B: int
            number of times to execute the generation loop
        """
        # Prepare R to be called from python
        R, R_functions = self._prepare_R()

        if method=="R":
            
            # Generate forbidden directions using a pre-defined list of rules
            forbidden_directions = self._generate_forbidden_directions()
            
            # Make a conversion of the variable names to avoid conflict when generating the Bayesian Network
            y_provisional, fd_provisional, columns_conversion, undo_conversion_dict = self._variable_name_conversion_and_forbidden_directions(microdata, forbidden_directions)
            
            # Build Bayesian Network
            bayesian_network = R_functions.BN_function(y_provisional, fd_provisional)
            
            # Extract bayesian network as matrix
            self.bn_result = np.array(R_functions.bnmat_function(bayesian_network))

            # Get adjacecncy matrix from the bayesian network
            adj_mat = pd.DataFrame(self.bn_result, columns=y_provisional.columns, index=y_provisional.columns)

            # Undo variable renaming
            adj_mat = adj_mat.rename(columns=undo_conversion_dict, index=undo_conversion_dict)

        else:
            # Build Bayesian Network using a pre-defined DAG
            dag_generator = DAG_generator(self.columns_spg)
            adj_mat = dag_generator.main()
        
        # Save adjacency matrix result
        #adj_mat.to_csv("ADJACENCY_MATRIX_new.csv")

        list(R_functions.topological_sort_function(adj_mat.to_numpy()))

        # Apply topological sort function
        # Order adj_mat matrix since adj_mat columns still having the same order as y
        topological_order = [adj_mat.columns[int(l-1)] for l in list(R_functions.topological_sort_function(adj_mat.to_numpy()))]

        # Order adjacency matrix according to the obtained topological sort
        adj_mat_s = adj_mat[topological_order].loc[topological_order]

        return topological_order, adj_mat_s
    

    def _compute_bootstrap_replicates(self, R_functions, microdata_area, mu_totals, number_of_farms):
        """
        Compute the number of bootstrap replicates or iterations for the empirical likelihood ratio test for the means.
        
        Parameters
        ----------
        R_functions: R enginer
            R_functions description
        microdata_area: pd.DataFrame
            microdata area description
        mu_totals: pd.DataFrame
            average values for the totals
        number_of_farms: int
            total number of farms in the population
        
        Returns
        ----------
        Bootstraps: int
            number of bootstrap replicates for the empirical likelihood
        weights_norm: weights_norm_type
            normalized weights for microdata rows
        """
        
        # Empirical likelihood ratio test for the means
        # weights is composed from
        #   {wts, nits}
        weights = R_functions.el_test_function(microdata_area, mu_totals.to_numpy())
        
        # Compute normalized weights for microdata rows
        # Use "wts" property from weights
        weights_norm = R_functions.normalize_weights(weights)

        # Get the smallest integer that is greather than or equal to the product of N_farms and the 
        # maximum value in the weights_norm vector.
        Bootstraps = math.ceil(R_functions.ceiling_wei1_function(weights_norm, number_of_farms))
        
        return Bootstraps, weights_norm
    

    def _prepare_R(self):
        """
        Method used to prepare R to be called from Python

        Parameters
        ----------

        Returns
        ----------
        R: rpy2.robjects.R
            object containing R instantiation to be called from python
        R_functions: rpy2.robjects.packages.Package
            R code containing some required functionalities
        """
        
        # Activate the functionalities to call a R function with pandas DataFrame
        pandas2ri.activate()

        # Instantiate R as a python object
        R = robjects.r

        # Load R function from script
        #with open("./src/R-scripts/R_functions.R") as r_file:
        with open(os.path.join(self.BASE_PATH, "./../../", "./src/R-scripts/R_functions.R")) as r_file:
            R_functions_string = r_file.read()

        # Convert text to R code
        R_functions = SignatureTranslatedAnonymousPackage(R_functions_string, "R_functions")

        return R, R_functions
    
    
    def _generate_forbidden_directions(self):
        """
        Method used to convert forbidden directions rules into a table of forbidden directions with the variables contained in the microdata.

        Parameters
        ----------

        Return
        ----------
        fd: pd.DataFrame
            forbidden directions
        """

        # Load rules to create forbidden directions
        rules = self.forbidden_directions_rules.copy(deep=True)
        
        # Add a column
        rules["remove"] = "no"

        # Create a dataframe to store the forbidden directions
        new_rules = pd.DataFrame(columns=["from", "to", "remove"])

        for i in rules.index:

            # Set a forbidden direction
            from_ = rules.at[i, "from"]
            to = rules.at[i, "to"]

            # Expand variables with the codes of the groups they belong if any
            from_list, from_expansion = self._expand_variable_with_codes(from_)
            to_list, to_expansion = self._expand_variable_with_codes(to)

            # If expansion has been carried out
            if from_expansion or to_expansion:
                rules.at[i, "remove"] = "yes"
                
                # Get all possible combinations for the given lists
                combinations = list(product(from_list, to_list))
                
                # Convert combinations into data frame
                rules_expanded = pd.DataFrame(combinations, columns=["from", "to"])
                rules_expanded["remove"] = "no"

                # Add new forbidden direction to existing ones
                new_rules = pd.concat([new_rules, rules_expanded], axis=0)
                
        fd = pd.concat([rules, new_rules])
        fd = fd[fd["remove"]=="no"].drop(columns=["remove"]).reset_index(drop=True)
                
        return fd
    
    
    def _expand_variable_with_codes(self, var):
        """
        Compose a list of variables using all possible codes for the given variable according to the group to which the variable belongs.
        Later the variables included in the list will be used to generate forbidden directions.

        Parameters
        ----------
        var: str
            variable name

        Returns
        ----------
        list_: list
            list of variables to be added to the forbidden directions
        expansion: bool
            whether the variable has been expanded with codes
        """
        # Set expansion with the common value for the first three cases
        expansion = True

        if var in self.crops_variables:
            list_ = [f"{code}.{var}" for code in self.crop_codes]
        elif var in self.animal_variables:
            list_ = [f"{code}.{var}" for code in self.animal_codes]
        elif var in self.subsidy_variables:
            list_ = [f"{code}.{var}" for code in self.subsidy_codes]
        # If the variable does not require expansion
        else:
            list_ = [var]
            expansion = False

        return list_, expansion


    def _variable_name_conversion_and_forbidden_directions(self, microdata, fd):
        """
        Make a conversion of the variables' names to avoid conflict "unrecognized node" when building the Bayesian Network.
        Use a easy nomenclature X.{i} to rename variables both for microdata as for forbidden directions. 

        Parameters
        ----------
        microdata: pd.DataFrame
            microdata with AGRICORE nomenclature
        fd: pd.DataFrame
            forbidden directions expanded

        Returns
        ----------
        y_provisional: pd.DataFrame
            microdata with a provisional nomenclature
        fd_provisional: pd.DataFrame
            microdata with a provisional nomenclatura
        conversion_dict: dict
            dictionary to perform the provisional naming for the adjacency matrix. {"X.{i}": "AGRICORE_NAME}
        undo_conversion_dict: dict
            dictionary to ease the undo of the provisional naming for the adjacency matrix. {"X.{i}": "AGRICORE_NAME}
        """

        # Make a copy of the original microdata object, average totals and area microdata
        y_provisional = microdata.copy(deep=True)
        
        # Make a copy of the original forbidden directions object
        fd_provisional = fd.copy(deep=True)
        
        # Create a dict that relates the standard name with the provissional name
        columns_conversion = dict([(var, f"X.{i}") for i, var in enumerate(y_provisional.columns.tolist())])
        
        # Rename columns in microdata
        y_provisional = y_provisional.rename(columns=columns_conversion)

        # Rename columns in the forbidden directions
        for col in ["from", "to"]:
            fd_provisional[col] = fd_provisional[col].apply(lambda x: columns_conversion[x])

        # Inver the order of the columns conversion dict to undo the variable renaming
        undo_conversion_dict = dict(zip(columns_conversion.values(), columns_conversion.keys()))

        return y_provisional, fd_provisional, columns_conversion, undo_conversion_dict
    

    def _manage_totals(self):
        """
        Method used to perform some operations to manage totals
        In the original code, the method works with the aggregated data of a NUTS-2 region.
        In addition, it contains for the NUTS-3 regions included within the NUTS-2 region
        the number of farms as well as some aggregated data for some crop (total area, crop
        production), animal and milk production variables.
        Notice that microdata available 

        Parameters
        ----------

        Returns
        ----------
        avg_area_by_farm_and_crop:  pd.DataFrame
            dataframe with the average total area for each crop
        number_of_farms: int
            number of farms contained in the totals dataset
        """
        
        # Load totals data
        totals = pd.read_csv(self.TOTALS_PATH, index_col=["Unnamed: 0"])

        # Select NUTS3 code
        totals = totals.loc[self.NUTS3].sum(axis=0).to_frame().transpose()

        # Load number of farms
        number_of_farms = pd.read_csv(self.NUMBER_OF_FARMS_PATH)[self.NUTS3].sum(axis=1).iloc[0]
        
        # Divide the totals of all the crops - total area by the total number of farms -> 
        # Average total area by farm and crop
        avg_area_and_quantity_by_farm_and_crop = totals.apply(lambda x: x/number_of_farms)

        return avg_area_and_quantity_by_farm_and_crop, number_of_farms
    

    def _generateFarmCode(self, categories, start=1):
        """
        Auxiliar functin to generate a unique farm code. The rule for the generation is to asign a single identifier for each instance.

        Parameters
        ----------
        categories: int
            number of categories to be asigned. In this case it is equal to the number of rows of the tabular data as the original file
            is organised as one instance per year and holding.

        Returns
        ----------
        farmCodes: list
            list containing the values to be asigned to the corresponding variable
        """

        # Set last category and number of points
        end = categories
        n_points = categories

        # Get number of categories
        result = np.linspace(start, end, n_points)

        # Convert result into integers
        farmCodes = [int(val) for val in result]

        return farmCodes
    

    def _correct_synthetic_population(self, y_synth):
        """
        Apply some corrections to the synthetic population generated

        Parameters
        ----------
        y_synth: dataframe
            synthetic population in DataFrame format
        
        Returns
        ----------
        y_synth_corr: dataframe
            synthetic population corrected
        """

        # Create a copy of the synthetic population
        y_synth_corr = y_synth.copy(deep=True)

        #----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Correction 1: crop variables, total area -> other crop related variables
        # If "cultivatedArea" is zero ->
        #        "irrigatedArea", 
        #        "quantitySold", 
        #        "valueSales", 
        #        "cropProduction", 
        #        "organicProductionType", 
        #        "variableCosts", 
        #        "landValue", 

        for code in self.crop_codes:
            for var in self.crops_variables:
                if var != "organicProductionType":
                    if var != "cultivatedArea":
                        y_synth_corr[f"{code}.{var}"] = y_synth_corr.apply(lambda row: row[f"{code}.{var}"] if row[f"{code}.cultivatedArea"] > 0 else 0.0, axis=1)

        # Correction 2: crop variables, irrigated area | total area -> irrigated area
        # Irrigated area must be <= cultivatedArea
        for code in self.crop_codes:
            y_synth_corr[f"{code}.irrigatedArea"] = y_synth_corr.apply(lambda row: row[f"{code}.irrigatedArea"] if row[f"{code}.cultivatedArea"] >= row[f"{code}.irrigatedArea"] else row[f"{code}.cultivatedArea"], axis=1)
                    

        # Correction 3: animal variables, number of animals -> other animal related variables
        # If "numberOfAnimals" is zero ->
        #       "numberOfAnimalsSold", "valueSoldAnimals", "numberAnimalsRearingBreading", "valueAnimalsRearingBreading", "numberAnimalsForSlaughtering", 
        #       "valueSlaughteredAnimals", "milkTotalProduction", "milkProductionSold", "milkTotalSales", "milkVariableCosts", "dairyCows", "variableCosts", 
        #       "woolTotalProduction", "woolProductionSold", "eggsTotalProduction", "eggsProductionSold", "eggsTotalSales", "manureTotalSales", "valueSold"
            
        for code in self.animal_codes:
            for var in self.animal_variables:
                if var != "numberOfAnimals":
                    y_synth_corr[f"{code}.{var}"] = y_synth_corr.apply(lambda row: row[f"{code}.{var}"] if row[f"{code}.numberOfAnimals"] > 0 else 0.0, axis=1)

        # Correction 4: animal variables, number of animals -> other animal number variables
        # The correction for numberOfAnimals is already done with the "Correction 3"
        # numberOfAnimals must be higher or equal than the number of animals for specific destinations:
        #        "numberOfAnimalsSold", "numberAnimalsRearingBreading", "numberAnimalsForSlaughtering", "dairyCows", 
                    
        if self.USE_CASE not in ["italy"]:
            for code in self.animal_codes:
                y_synth_corr[f"{code}.{var}"] = y_synth_corr.apply(lambda row: row[f"{code}.numberOfAnimals"] if row[f"{code}.numberOfAnimalsSold"] + row[f"{code}.numberAnimalsRearingBreading"] + row[f"{code}.numberAnimalsForSlaughtering"] + row[f"{code}.dairyCows"] <= row[f"{code}.numberOfAnimals"] else row[f"{code}.numberOfAnimalsSold"] + row[f"{code}.numberAnimalsRearingBreading"] + row[f"{code}.numberAnimalsForSlaughtering"] + row[f"{code}.dairyCows"], axis=1)

            if False:
                # Correction 5: farmCode
                y_synth_corr["farmCode"] = self._generateFarmCode(y_synth_corr.shape[0])


            # Correction 6: age
            # Age correction encompasses replace zero values by logical values and rounding the value to the nearest integer
            mean = 55.84347826086957
            std = 11.450945010239021
            y_synth_corr["holderAge"] = y_synth_corr["holderAge"].apply(lambda x: int(x) if int(x) > 0 else int(np.random.normal(mean, std, 1)[0]))
            y_synth_corr["holderAge"] = y_synth_corr["holderAge"].apply(lambda x: x if x >= 18 else 18)

            # Standarsise data type
            y_synth_corr["holderAge"] = y_synth_corr["holderAge"].astype(int)
            
            # Correction 7: holderSuccessors
            #y_synth_corr["holderSuccessors"] = pd.DataFrame(np.random.lognormal(0.5, 0.5, y_synth_corr.shape[0])).apply(lambda x: int(x), axis=1)
            proba_dict = {0: 0.9, 1: 0.1}
            y_synth_corr["holderSuccessors"] = pd.DataFrame(np.random.choice(list(proba_dict.keys()), y_synth_corr.shape[0], p=list(proba_dict.values())))

            # Correction 8: holderSuccessorsAge
            #if y_synth_corr["holderSuccessorsAge"].nunique()==1:
            min_holderAge = y_synth_corr["holderAge"].min()
            #y_synth_corr["holderSuccessorsAge"] = y_synth_corr["holderSuccessors"].apply(lambda x: x if x==0 else int(np.random.uniform(0, 35, 1)[0]))
            y_synth_corr["holderSuccessorsAge"] = y_synth_corr.apply(lambda x: int(0) if x["holderSuccessors"]==0 else \
                                                                            max(0, int(x["holderAge"] - np.random.uniform(min_holderAge, max(x["holderAge"], 45)))
                                                                            ), axis=1)
            
            y_synth_corr["holderSuccessorsAge"] = y_synth_corr["holderSuccessorsAge"].astype(int)
            
            # Correction 9: holderFamilyMembers
            #y_synth_corr["holderFamilyMembers"] = y_synth_corr["holderSuccessors"].apply(lambda x: int(x + 2))
            proba_dict = {0: 0.55, 1: 0.31, 2: 0.11, 3: 0.01, 4: 0.01, 5: 0.01, }
            y_synth_corr["holderFamilyMembers"] = y_synth_corr.apply(
                lambda x: int(1 + x["holderSuccessors"] + int(np.random.choice(list(proba_dict.keys()), 1, p=list(proba_dict.values())))), axis=1)
            
            # Correction 10: holderGender
            y_synth_corr["holderGender"] = y_synth_corr["holderGender"].apply(lambda x: int(float(x)) if int(float(x)) in [1, 2] else int(np.random.uniform(1, 3, 1)[0]))

            # Correction 11: sellingPrice crops
            for agg in self.crop_codes:
                y_synth_corr[f"{agg}.sellingPrice"] = y_synth_corr.apply(lambda x: x[f"{agg}.valueSales"] / x[f"{agg}.quantitySold"] if x[f"{agg}.quantitySold"] > 0 else 0, axis=1)

                # Get average value for sellingPrice
                avg_sellingPrice = y_synth_corr[y_synth_corr[f"{agg}.sellingPrice"]>0][f"{agg}.sellingPrice"].mean()

                # Get indexes of zero value
                indexes_sellingPrice = y_synth_corr[y_synth_corr[f"{agg}.sellingPrice"]==0].index

                # Input mean value
                y_synth_corr.loc[indexes_sellingPrice, f"{agg}.sellingPrice"] = np.ones(len(indexes_sellingPrice))*avg_sellingPrice
        
            # Correction 12: sellingPrice animals This does not exists in the SP as it is computed internally in the ABM
            
            # Correction 13: Variable Costs per produced unit (CV - [€/ton])
            # Read variableCostsCrops data
            variableCostsCrop_data = pd.read_csv(self.VARIABLE_COSTS_CROP_FILEPATH)

            for crop in variableCostsCrop_data["Crop"].unique():
                variableCostsCrops = variableCostsCrop_data[variableCostsCrop_data["Crop"]==crop]["variableCostsCrops [€/Ton]"].item()    
                y_synth_corr[f"{crop}.variableCostsCrops"] = y_synth_corr.apply(lambda x: variableCostsCrops, axis=1)

            # Correction 14: milkVariableCosts [€/Ton]
            if y_synth_corr["DAIRY.milkVariableCosts"].nunique()==1:

                milkVariableCosts_dict = {"2014": 371.2,
                                        "2015": 306.9,
                                        "2016": 284.6,
                                        "2017": 248.5,
                                        "2018": 341.1,
                                        "2019": 344.3,
                                        "2020": 341.1,}
                
                y_synth_corr["DAIRY.milkVariableCosts"] = y_synth_corr.apply(lambda x: milkVariableCosts_dict[self.YEAR] if x["DAIRY.numberOfAnimals"] >  0 else 0.0, axis=1) 
        
            # Correction 15: rentBalance [€] & rentBalanceArea [Ha]
            y_synth_corr["rentBalanceArea"] = y_synth_corr.apply(lambda x: x["rentBalanceIn"] - x["rentBalanceOut"], axis=1)
            # rentBalance [€] = rentPaid [€/Ha] * rentBalanceArea [Ha]
            y_synth_corr["rentBalance"] = y_synth_corr.apply(lambda x: x["rentPaid"]*(x["rentBalanceIn"] - x["rentBalanceOut"]), axis=1)
            #y_synth_corr = y_synth_corr.drop(columns=["rentBalanceIn", "rentBalanceOut"])


            # Correction 16: technoEconomicalOrientation
            if y_synth_corr["technicalEconomicOrientation"].unique()[0] == 'A_TY_90_TF':
                y_synth_corr["technicalEconomicOrientation"] = y_synth_corr["technicalEconomicOrientation"].apply(lambda x: 0)

            # Correction 17: altitude
            if y_synth_corr["altitude"].unique()[0] == 0.0:
                y_synth_corr["altitude"] = y_synth_corr["altitude"].apply(lambda x: 1)

            # Correction 18: sellingPrice
            

        return y_synth_corr
        

    def _check_sp_godness(self, original_data, synthetic_data, p_value=0.05):
        """
        Method to check the goodness of the synthetic population
        
        Parameters
        ---------- 
        original_data: pd.DataFrame 
            original dataset from which the synthetic population is generated
        synthetic_population: pd.DataFrame 
            synthetic population generated using the bayesian network approach
        p_value: float
            determine whether there is a statistical significance between th

        Returns
        ----------
        p_values: np.narray
            an array of p_values per variables 
        """

        # Create a dataframe to store the results for each variable
        t_test_results = pd.DataFrame(columns =['variable', 'p_value'])

        # Perform independent two sample t-test
        for c in [cc for cc in synthetic_data.columns.tolist() if cc not in self.categoricals]:
            
            # Perform t-test
            t, p = ttest_ind(original_data[c], synthetic_data[c])
            
            if (p < p_value):
                t_test_results.loc[len(t_test_results)] = [c, p]

        return t_test_results
    

    def compare_mean_and_std(R_result, P_result):
        """
        Compare the mean and standard deviation of both synthetic populations
        """

        for c in P_result.columns:
            print(f"{c}")
            print(f"{round(R_result[c].mean(), 2)} : {round(P_result[c].mean(), 2)}")
            print(f"{round(R_result[c].std(), 2)} : {round(P_result[c].std(), 2)}")
            print("-------------------------------------------------")
        
    
    def _generate_synthetic_pool(self, microdata, topological_order, adj_mat_s, columns_spg, categoricals, encoder):
        """
        Generate a synthetic pool using the Bayesian Network

        microdata: pd.DataFrame
            original microdata sample
        Bootstraps: int
            Number of iterations required to guarantee that the most representative farm is completely represented in the synthetic population
        topological_order: pd.DataFrame
            
        adj_mat_s: np.array
            adjacency matrix sorted
        columns_spg: list
            variables included in the synthetic population
        n_rows: int
            number of farms in the original sample
        categoricals: list
            categorical variables defined in the synthetic population
        encoder: OrdinalEncoder
            object encoder defined to encode categorical variables
        """
        t0 = time.time()

        synthetic_sample = _generate_synthetic_sample(microdata[topological_order], adj_mat_s, columns_spg, categoricals)
        synthetic_sample = _decode_data(synthetic_sample, categoricals, encoder)
        
        t1 = time.time()

        print(f"Generation time: {t1 - t0}")

        return synthetic_sample
    
    
    def _generate_synthetic_population(self, synthetic_pool, Bootstraps, weights_norm, number_of_farms):
        """
        Sample synthetic pool according to the likelyhood of each farm to be part of the totals mean.
        With this sampling process, a synthetic population is created keeping the totals relationship.

        Parameters
        ----------
        synthetic_pool: pd.DataFrame
            synthetic pool
        Bootstraps: int
            number of iterations performed to crete the synthetic pool
        weights_norm: np.array
            normalised weights of each farm (rows) to belong to the population according to totals
        number_of_farms: int
            number of farms of the synthetic population
        Returns
        ----------
        synthetic_population: pd.DataFrame
            synthetic population
        """

        # Generate a matrix with the indexes of the farms.
        # Each row contains the indexes for all the farms that represent a specific farm in the original sample
        pinakas = np.linspace(0, Bootstraps*self.n_rows-1, Bootstraps*self.n_rows).reshape(Bootstraps, self.n_rows).transpose()
        
        # Initialise synethetic population
        synthetic_population = pd.DataFrame()

        n_farms_counter = 0

        # Compose row by row
        for i in range(self.n_rows):
            
            # Get row weight
            farm_weight = weights_norm[i]

            # Compute the number of farms given the likelihood of each farm
            n_farms = math.ceil(farm_weight*number_of_farms)

            n_farms_counter += n_farms

            # Sampe randomly items from the row i
            selection = [int(x) for x in random.sample(pinakas[i, :].tolist(), k=n_farms)]
            
            # Add pool selection to the synthetic population
            synthetic_population = pd.concat([synthetic_population, synthetic_pool.loc[selection, :]])
            
        print(f"REAL NUMBER OF FARMS GENERATED: {n_farms_counter}")
        
        synthetic_population = synthetic_population.reset_index(drop=True)

        return synthetic_population
    
    
    def _assign_district_value(self, synthetic_population):
        """
        Assign district value based on experimental likelihood

        Parameters
        ----------
        sp: pd.DataFrame
            synthetic population
        """

        if self.USE_CASE=="andalusia":
            rla = RegionLevelSpain(self.BASE_PATH, self.USE_CASE, self.YEAR)
            synthetic_population = rla._fill_regionLevel(synthetic_population)

        elif self.USE_CASE=="italy":
            pass
        elif self.USE_CASE=="greece":
            rlg = RegionLevelGreece()
            synthetic_population = rlg._fill_regionLevel(synthetic_population)
        elif self.USE_CASE=="poland":
            rlp = RegionLevelPoland()
            synthetic_population = rlp._fill_regionLevel(synthetic_population)
        else:
            pass
            
        return synthetic_population
    

    def _store_synthetic_population_fidelity(self, synthetic_population):
        """
        Store synthetic population fidelity with regard original population at NUTS2 and NUTS3 levels
        """

        # Get the list of variables used to compute totals
        totals_variables = [c for c in synthetic_population.columns for var in self.totals_variables if c.endswith(var)]

        # Load totals data and select NUTS3
        totals = pd.read_csv(self.TOTALS_PATH, index_col=["Unnamed: 0"])

        for var in ["regionLevel1", "regionLevel2", ]:
            self._compute_ratios(synthetic_population, totals, totals_variables, var)


    def _compute_ratios(self, synthetic_population, totals, totals_variables, var):
        """
        
        """
        
        for nuts in totals[var].unique():
            
            # Stack synthetic population totals and farms totals
            totals_nuts = pd.concat([
                totals[totals[var]==nuts].sum().to_frame().transpose().rename(index={0: "original_population"})[totals_variables], 
                synthetic_population[synthetic_population[var]==nuts].sum().to_frame().transpose().rename(index={0: "synthetic_population"})[totals_variables]])

            # Compute ratio between synthetic population and real population
            eps = 1e-6
            totals_ratio_nuts = totals_nuts.apply(lambda x: (x["synthetic_population"] + eps) / (x["original_population"] + eps)).to_frame().transpose().rename(index={0: "ratio"})

            totals_nuts = pd.concat([totals_nuts, totals_ratio_nuts], axis=0)
            
            # Save results
            totals_ratio_nuts.to_csv(os.path.join(self.RESULTS_PATH, f"SP_results_{self.USE_CASE}_{nuts}_{self.YEAR}.csv"), index=False)
            

    def main(self):
        """
        Main method of the SPG class

        Parameters
        ----------
        microdata: pd.DataFrame
        """

        # Load microdata and convert FADN nomenclature into AGRICORE nomenclature. 
        # Preprocess microdata to get data ready to build the Bayessian NetworkCows
        microdata, weights = self._load_and_preprocess_microdata()
        
        #weights.to_csv(f"weights-{self.USE_CASE}-{self.YEAR}.csv")
        
        # Use R to build and fit the bayesian network
        topological_order, adj_mat_s = self._build_bayesian_network(microdata)
        
        # Upsample data
        microdata = microdata.reindex(microdata.index.repeat(weights.to_numpy().flatten())).reset_index(drop=True)

        # Disable TTL for dask to avoid timeout during long executions
        cfg.set({'distributed.scheduler.worker-ttl': None})
        
        # Genearte synthetic pool
        synthetic_population = self._generate_synthetic_pool(microdata, 
                                                             topological_order, 
                                                             adj_mat_s, 
                                                             self.columns_spg, 
                                                             self.categoricals, 
                                                             self.encoder)
        
        # Assign Region
        synthetic_population = self._assign_district_value(synthetic_population)

        # Correct synthetic population
        synthetic_population = self._correct_synthetic_population(synthetic_population)

        # Save the result in a csv to further use
        date = datetime.now()
        
        try:
            csv_path = os.path.join(self.SYNTHETIC_POPULATION_PATH, f'Synthetic-Population-{self.USE_CASE}-{self.YEAR}-{date.month}-{date.day}-{date.hour}-{date.minute}.csv')
            synthetic_population.to_csv(csv_path, index=False)
            print(f"Synthetic Population saved on '{csv_path}'")
        except:
            print("Synthetic population was not saved properly")

        return synthetic_population
    