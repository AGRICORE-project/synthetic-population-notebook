import pandas as pd
import os
import numpy as np


class DAG_generator():
    def __init__(self, columns_spg):
        
        self.columns_spg = columns_spg
        
        self.DAG = pd.DataFrame(np.zeros((len(self.columns_spg), len(self.columns_spg))), index=self.columns_spg, columns=self.columns_spg)

        self.Crops = [
            "quantitySold", 
            "valueSales", 
            "cropProduction", 
            "irrigatedArea", 
            "cultivatedArea", 
            "organicProductionType", 
            "variableCostsCrops", 
            "landValue", 
            "quantityUsed", 
            "sellingPrice"]

        self.Animals = [
            "numberOfAnimals", 
            "numberOfAnimalsSold", 
            "valueSoldAnimals", 
            "numberAnimalsRearingBreading", 
            "valueAnimalsRearingBreading", 
            "numberAnimalsForSlaughtering", 
            "valueSlaughteredAnimals", 
            "milkTotalProduction", 
            "milkProductionSold", 
            "milkTotalSales", 
            "milkVariableCosts", 
            "dairyCows", 
            "variableCostsAnimals", 
            "woolTotalProduction", 
            "woolProductionSold", 
            "eggsTotalProduction", 
            "eggsProductionSold", 
            "eggsTotalSales", 
            "manureTotalSales", ]
            
        self.Subsidies = [
            "policyIdentifier", 
            "policyDescription", 
            "isCoupled", 
            "value",
        ]

        self.Farm = [
            "lat", 
            "long", 
            "altitude", 
            "farmCode", 
            "technicalEconomicOrientation", 
            "weight_ra", 
            "regionLevel1Name", 
            "regionLevel2Name", 
            "regionLevel3Name", 
            "regionLevel1", 
            "regionLevel2", 
            "regionLevel3", 
            "weight_reg", ]

        self.Economic = [
            "agriculturalLandArea",
            "agriculturalLandValue",
            "agriculturalLandHectaresAdquisition",
            "landImprovements", 
            "forestLandArea",
            "forestLandValue",
            "farmBuildingsValue", 
            "machineryAndEquipment", 
            "intangibleAssetsTradable", 
            "intangibleAssetsNonTradable", 
            "otherNonCurrentAssets", 
            "longAndMediumTermLoans", 
            "totalCurrentAssets", 
            "farmNetIncome", 
            "grossFarmIncome", 
            "subsidiesOnInvestments", 
            "vatBalanceOnInvestments", 
            "totalOutputCropsAndCropProduction", 
            "totalOutputLivestockAndLivestockProduction", 
            "otherOutputs", 
            "totalIntermediateConsumption", 
            "taxes", 
            "vatBalanceExcludingInvestments", 
            "fixedAssets", 
            "depreciation", 
            "totalExternalFactors", 
            "machinery",
            "rentPaid", 
            "rentBalance", 
            "specificCropCosts",
            "plantationsValue",]

        self.Holder = [
            "holderAge", 
            "holderGender", 
            "holderSuccessors", 
            "holderSuccessorsAge", 
            "holderFamilyMembers", 
            "yearNumber", ]


        self.Economic_dict = {
            # 1. Current Assets
            "totalCurrentAssets": ["fixedAssets", ], 
                "grossFarmIncome": ["totalCurrentAssets", ], 
                    "farmNetIncome": ["grossFarmIncome"], 
                        "totalOutputCropsAndCropProduction": ["farmNetIncome"], 
                            "totalIntermediateConsumption": ["totalOutputCropsAndCropProduction"], 
                        "totalOutputLivestockAndLivestockProduction": ["farmNetIncome"], 
                        "subsidiesOnInvestments": ["farmNetIncome"], 
                "intangibleAssetsTradable": ["totalCurrentAssets", ], 
                "vatBalanceOnInvestments": ["totalCurrentAssets", ], 
                "otherOutputs": ["totalCurrentAssets", ], 
                "taxes": ["totalCurrentAssets", ], 
                "vatBalanceExcludingInvestments": ["totalCurrentAssets", ], 
                "specificCropCosts": ["cultivatedArea", "irrigatedArea"],
                "plantationsValue":  ["cultivatedArea", "irrigatedArea"], 
             

            # 2. Fixed Assets
            "agriculturalLandArea": [], #"technicalEconomicOrientation", "regionLevel2", 
                "agriculturalLandHectaresAdquisition": ["agriculturalLandArea"], 
                "fixedAssets": ["agriculturalLandArea", ], #"technicalEconomicOrientation", "regionLevel2", 

                    "agriculturalLandValue": ["fixedAssets", "agriculturalLandArea"], 
                        "landImprovements": ["agriculturalLandValue"], 

                    "forestLandValue": ["fixedAssets", ], 
                    "farmBuildingsValue": ["fixedAssets", ], 

                    "forestLandArea": ["fixedAssets", ], 

                    "machineryAndEquipment": ["fixedAssets", ],
                    "machinery": ["fixedAssets", ],  
                    "intangibleAssetsNonTradable": ["fixedAssets", ], 

                    "otherNonCurrentAssets": ["fixedAssets", ], 
                    "depreciation": ["fixedAssets", ], 
                    "totalExternalFactors": ["fixedAssets", ], 
                    "rentBalance": ["fixedAssets", "cultivatedArea", "irrigatedArea"], 

            # 3. Liabilities
            "longAndMediumTermLoans": ["fixedAssets"], 

            # 4. Equity
        }
        
        #   Son: Parents
        self.Crops_dict = {
            "cultivatedArea":        [], #"agriculturalLandArea"
                "irrigatedArea":         ["cultivatedArea", ], 
                    "cropProduction":        ["cultivatedArea", "irrigatedArea", ], 
                        "quantitySold":          ["cropProduction", ], 
                            "quantityUsed":          ["cropProduction", "quantitySold", ], 
                            "valueSales":            ["quantitySold"], 
                    "variableCostsCrops":    ["cultivatedArea", "irrigatedArea", ], 
                "landValue":             ["cultivatedArea", "irrigatedArea", ], 
            "organicProductionType": [], # Categorical
                "sellingPrice":          [], 
        }

        self.Animals_dict = {
            'numberOfAnimals':              [ ],
                'numberOfAnimalsSold':          ["numberOfAnimals", ],
                    'valueSoldAnimals':             ["numberOfAnimalsSold", ],
                'numberAnimalsRearingBreading': ["numberOfAnimals", ],
                    'valueAnimalsRearingBreading':  ["numberAnimalsRearingBreading", ],
                'numberAnimalsForSlaughtering': ["numberOfAnimals", ],
                    'valueSlaughteredAnimals':      ["numberAnimalsForSlaughtering", ],
            'dairyCows':                    ["numberOfAnimals",  ],    
                'milkTotalProduction':          ["dairyCows", ],
                    'milkProductionSold':           ["milkTotalProduction", ], 
                        'milkTotalSales':               ["milkProductionSold", ],
                'milkVariableCosts':            ["dairyCows", ],
            'variableCostsAnimals':         ["numberOfAnimals", "numberAnimalsRearingBreading", "numberAnimalsForSlaughtering", "dairyCows", ],
            'woolTotalProduction':          ["numberOfAnimals"],
                'woolProductionSold':           ["woolTotalProduction", ],
            'eggsTotalProduction':          ["numberOfAnimals", ],
                'eggsProductionSold':           ["eggsTotalProduction", ],
                    'eggsTotalSales':               ["eggsProductionSold", ],
            'manureTotalSales':             ["numberOfAnimals", ],
            #'valueSold':                    ["valueSoldAnimals", "valueAnimalsRearingBreading", "valueSlaughteredAnimals", "milkProductionSold", "milkTotalSales", "woolProductionSold", "eggsProductionSold", "eggsTotalSales", "manureTotalSales", "valueSold", ]
            }


        self.Subsidies_dict = {
            "policyIdentifier": [], 
            "policyDescription": [], 
            "isCoupled": [], 
            "value": [], #["agriculturalLandArea", "cultivatedArea", 'numberOfAnimals', "fixedAssets", ], 
        }


    def main(self):

        for son in self.DAG.columns:
            
            if "." in son:
                gen_son = son[son.index(".")+1:]
                
                # Crops
                if gen_son in self.Crops_dict.keys():
                    crop = son[:son.index(".")]
                    
                    for gen_parent in self.Crops_dict[gen_son]:
                        if gen_parent in list(self.Crops_dict.keys()):
                            parent = f"{crop}.{gen_parent}"
                        else:
                            parent = gen_parent

                        self.DAG.at[parent, son] = 1

                # Animals
                elif gen_son in self.Animals_dict.keys():
                    animal = son[:son.index(".")]
                    
                    for gen_parent in self.Animals_dict[gen_son]:
                        if gen_parent in list(self.Animals_dict.keys()):
                            parent = f"{animal}.{gen_parent}"
                            
                        else:
                            parent = gen_parent
                            
                        self.DAG.at[parent, son] = 1

                elif gen_son in self.Subsidies_dict.keys():
                    
                    # Pick all crops and animals
                    # Expand parent variables for crops and animals

                    # Get list of crops and animals available
                    crop_codes = [var.replace(".cultivatedArea", "") for var in self.DAG.columns if var.endswith(".cultivatedArea")]
                    animal_codes = [var.replace(".numberOfAnimals", "") for var in self.DAG.columns if var.endswith(".numberOfAnimals")]

                    # Compose the list of parent variables
                    parent_list = []

                    for gen_parent in self.Subsidies_dict[gen_son]:
                        
                        if gen_parent in self.Crops:
                            for crop in crop_codes:
                                parent_list.append(f"{crop}.{gen_parent}")

                        elif gen_parent in self.Animals:
                            for animal in animal_codes:
                                parent_list.append(f"{animal}.{gen_parent}")
                        else:
                            parent_list.append(gen_parent)

                    for parent in parent_list:
                        self.DAG.at[parent, son] = 1
                    

            else:
                # Economic variables
                if son in self.Economic_dict.keys():
                    for gen_parent in self.Economic_dict[son]:
                        if gen_parent in self.Economic_dict.keys():
                            self.DAG.at[gen_parent, son] = 1
                        elif gen_parent in self.Crops_dict.keys():
                            crop_codes = [var.replace(".cultivatedArea", "") for var in self.DAG.columns if var.endswith(".cultivatedArea")]
                            for crop in crop_codes:
                                parent = f"{crop}.{gen_parent}"
                                self.DAG.at[parent, son] = 1
                        else:
                            pass
                        
        return self.DAG
