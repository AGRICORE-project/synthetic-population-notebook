import pandas as pd
import numpy as np
import os
from src.config import NUTS2_conversion


class WeightComputer_old():
    def __init__(self, METADATA_PATH, USE_CASE, YEAR):

        # Load metadata required to compute weights
        METADATA_FILE = f"FADN_metadata_{USE_CASE}.xlsx"
        self.metadata = pd.read_excel(os.path.join(METADATA_PATH, METADATA_FILE))
        
        # Process metadata file
        self.metadata["(SYS02) Farms represented (nb)"] = self.metadata["(SYS02) Farms represented (nb)"].astype(int)
        self.metadata["Year"] = self.metadata["Year"].astype(int)

        # Declare encoding Economic Size variable according to metadata format
        es_naming = {'(2) 8 000 - < 25 000 EUR': 1, 
                    '(3) 25 000 - < 50 000 EUR': 2,
                    '(4) 50 000 - < 100 000 EUR': 3, 
                    '(5) 100 000 - < 500 000 EUR': 4,
                    '(6) >= 500 000 EUR': 5}
        
        # Encode variable
        self.metadata["Economic Size"] = self.metadata["Economic Size"].apply(lambda x: es_naming[x])

        # Convert unit from k€ to €
        self.metadata["(SE005) Economic size (€'000)"] = self.metadata["(SE005) Economic size (€'000)"].apply(lambda x: 1000*x)
        self.metadata = self.metadata.rename(columns={"(SE005) Economic size (€'000)": "(SE005) Economic size (€)"})

        # Update region value
        self.metadata["Region"] = self.metadata["Region"].apply(lambda x: NUTS2_conversion[x])
        self.metadata = self.metadata.rename(columns={"Region": "A_LO_40_N2", 
                                                      "Year": "YEAR", 
                                                      })

        # Select year
        self.metadata = self.metadata[self.metadata["YEAR"]==int(YEAR)]
        
        self.index_columns = ["YEAR", "Member State", "A_LO_40_N2", "Economic Size"]
        self.metadata = self.metadata.rename(columns={c: c[1:6] for c in self.metadata.columns if c not in self.index_columns})
        self.value_coulmns = [c for c in self.metadata.columns if c not in self.index_columns]

        # Define economic sizes relationship between A_TY_80_W and SE005
        # From, less than
        self.es_ranges = pd.DataFrame({
             1: [      0,     2000, 0], 
             2: [   2000,     4000, 0], 
             3: [   4000,     8000, 0], 
             4: [   8000,    15000, 1], 
             5: [  15000,    25000, 1], 
             6: [  25000,    50000, 2], 
             7: [  50000,   100000, 3], 
             8: [ 100000,   250000, 4], 
             9: [ 250000,   500000, 4], 
            10: [ 500000,   750000, 5], 
            11: [ 750000,  1000000, 5], 
            12: [1000000,  1500000, 5], 
            13: [1500000,  3000000, 5], 
            14: [3000000, 99000000, 5], }).transpose().rename(columns={0: "from", 1: "less than", 2: "Economic Size"})
        
        self.es_ranges["less than"] = self.es_ranges.apply(lambda x: x["less than"] - 1e-3, axis=1)

        # display(self.metadata)
        # display(self.metadata["SYS02"].sum())


    def main(self, microdata, var):
        
        n_rows = microdata.shape[0]

        # Missing NUTS3 value
        # Assign unique NUTS3 value
        if var=="A_LO_40_N":
            default_NUTS3=var
            new_var_df = pd.DataFrame([default_NUTS3 for _ in range(n_rows)], columns=[var])
            microdata = pd.concat([microdata, new_var_df], axis=1)

            return microdata
        
        elif var=="A_LO_40_N2":
            default_NUTS2=var
            new_var_df = pd.DataFrame([default_NUTS2 for _ in range(n_rows)], columns=[var])
            microdata = pd.concat([microdata, new_var_df], axis=1)

            return microdata
            
        # Techno-economical orientation
        # Assign unique techno-economical orientation
        elif var=="A_TY_90_TF":
            default_OTE=var
            new_var_df = pd.DataFrame([default_OTE for _ in range(n_rows)], columns=[var])
            microdata = pd.concat([microdata, new_var_df], axis=1)

            return microdata

        # Economic size
        elif var=="A_TY_90_ES":
            
            if "SE005" in microdata.columns:
                
                # Convert SE005 units k€ -> €
                microdata["SE005"] = microdata["SE005"].apply(lambda x: 1000*x)

                # Bucketize data
                microdata["SE005 bin"] = pd.cut(
                    microdata["SE005"], 
                    bins=self.es_ranges["from"].tolist() + [self.es_ranges["less than"].tolist()[-1]], 
                    labels=self.es_ranges["Economic Size"], 
                    ordered=False)
                
                # Equivalence between A_TY_90_ES and SE005 bin
                # Not necessary as A_TY_90_ES is only used for weights computing/extrapolation
                microdata["A_TY_90_ES"] = microdata.apply(lambda x: x["SE005 bin"], axis=1)

                return microdata
            
            else:
                for var in ["A_TY_90_ES", "SE005", "SE005 bin"]:
                    microdata[var] = pd.DataFrame(np.zeros((n_rows, 1)), columns=var)

                return microdata

        # Weights
        elif var=="A_TY_80_W":
            
            # Generate weights by NUTS2. 
            # Data for NUTS3 resolution level is only available for andalusian use case
            for NUTS2 in microdata["A_LO_40_N2"].unique():
                for es in microdata["SE005 bin"].unique():
                    
                    mask = (microdata["SE005 bin"]==es)&(microdata["A_LO_40_N2"]==NUTS2)

                    sel = microdata[mask]
                    sel_meta = self.metadata[(self.metadata["Economic Size"]==es)&(self.metadata["A_LO_40_N2"]==NUTS2)]

                    if sel.shape[0]>0 and sel_meta.shape[0]>0:
                        
                        # Weight parameters
                        number_of_farms_represented = sel_meta["SYS02"].item()
                        number_of_farms_sample = sel.shape[0]
                        
                        # Comute ratio real population vs sample size
                        weight = number_of_farms_represented/number_of_farms_sample
                        
                        # Assign weight
                        microdata.loc[mask, "A_TY_80_W"] = weight
                    
            return microdata

        else:
            pass


class WeightComputer():
    def __init__(self, METADATA_PATH, USE_CASE, YEAR):
        
        self.USE_CASE = USE_CASE

        # Load metadata required to compute weights
        METADATA_FILE = f"FADN_metadata_{self.USE_CASE}.csv"
        
        metadata = pd.read_csv(os.path.join(METADATA_PATH, METADATA_FILE), usecols=["so_eur", "uaarea", "unit", "TIME_PERIOD", "OBS_VALUE", ]).rename(columns={
            "so_eur": "Economic Size", 
            "uaarea": "UAArea", 
            "TIME_PERIOD": "YEAR", 
            "OBS_VALUE": "value", 
        })

        # Select year according to the closest year in the metadata
        self.YEAR = min(sorted(metadata["YEAR"].unique()), key=lambda x:abs(x-int(YEAR)))

        self.es_dict = {
            'KE0':         [-1e3,    0], 
            'KE_GT0_LT2':  [   0,    2], 
            'KE2-3':       [   2,    4], 
            'KE4-7':       [   4,    8], 
            'KE8-14':      [   8,   15], 
            'KE15-24':     [  15,   25], 
            'KE25-49':     [  25,   50], 
            'KE50-99':     [  50,  100], 
            'KE100-249':   [ 100,  250], 
            'KE250-499':   [ 250,  500],
            'KE_GE500':    [ 500,  1e6], 
            'TOTAL':       [-1e3, 1e6]}

        self.uaa_dict = {
            'HA0':          [-1e3,   0], 
            'HA_GT0_LT2':   [   0,   2], 
            'HA2-4':        [   2,   5], 
            'HA5-9':        [   5,  10],
            'HA10-19':      [  10,  20], 
            'HA20-29':      [  20,  30], 
            'HA30-49':      [  30,  50], 
            'HA50-99':      [  50, 100], 
            'HA_GE100':     [ 100, 1e6], 
            'TOTAL':        [-1e3, 1e6], 
        }


        metadata["ge ES"] = metadata.apply(lambda x: self.es_dict[x["Economic Size"]][0], axis=1)
        metadata["l ES"]  = metadata.apply(lambda x: self.es_dict[x["Economic Size"]][1], axis=1)

        metadata["ge UAA"] = metadata.apply(lambda x: self.uaa_dict[x["UAArea"]][0], axis=1)
        metadata["l UAA"]  = metadata.apply(lambda x: self.uaa_dict[x["UAArea"]][1], axis=1)

        # Filter data
        metadata = metadata[~metadata["value"].isna()]
        metadata = metadata[(metadata["Economic Size"]!="TOTAL")&(metadata["UAArea"]!="TOTAL")]

        metadata = metadata.fillna(0)

        es_cats   = {cat_old: cat for cat, cat_old in enumerate(self.es_dict.keys())}
        uaa_cats  = {cat_old: cat for cat, cat_old in enumerate(self.uaa_dict.keys())}

        metadata["ES"]  = metadata.apply(lambda x: es_cats[x["Economic Size"]], axis=1)
        metadata["UAA"] = metadata.apply(lambda x: uaa_cats[str((x["UAArea"]))], axis=1)

        metadata = metadata.drop(columns=["Economic Size", "UAArea"])

        self.metadata = metadata.copy(deep=True)


    def main(self, microdata, var):
        
        n_rows = microdata.shape[0]

        # Missing NUTS3 value
        # Assign unique NUTS3 value
        if var=="A_LO_40_N":
            default_NUTS3=var
            new_var_df = pd.DataFrame([default_NUTS3 for _ in range(n_rows)], columns=[var])
            microdata = pd.concat([microdata, new_var_df], axis=1)

            return microdata
        
        elif var=="A_LO_40_N2":
            default_NUTS2=var
            new_var_df = pd.DataFrame([default_NUTS2 for _ in range(n_rows)], columns=[var])
            microdata = pd.concat([microdata, new_var_df], axis=1)

            return microdata
            
        # Techno-economical orientation
        # Assign unique techno-economical orientation
        elif var=="A_TY_90_TF":
            default_OTE=var
            new_var_df = pd.DataFrame([default_OTE for _ in range(n_rows)], columns=[var])
            microdata = pd.concat([microdata, new_var_df], axis=1)

            return microdata

        # Economic size
        elif var=="A_TY_90_ES":
            
            if "SE005" in microdata.columns:
                
                microdata[var] = microdata.apply(lambda x: self.metadata[(x["SE005"] >= self.metadata["ge ES"])&
                                                                         (x["SE005"] <  self.metadata["l ES"])]["ES"].unique()[0], axis=1)

                return microdata
            
            else:
                for var in ["A_TY_90_ES", "SE005", ]:
                    microdata[var] = pd.DataFrame(np.zeros((n_rows, 1)), columns=[var])

                return microdata

        # Weights
        elif var=="A_TY_80_W":


            NUTS2_var = "A_LO_40_N2" # [NUTS2]
            UAA_var =        "SE025" # [Ha]
            ES_var =         "SE005" # [k€]


            # Create a copy of the microdata
            weight_df = microdata[[NUTS2_var, UAA_var, ES_var]].copy(deep=True).rename(columns={UAA_var: "UAA value", ES_var: "ES value"})

            # Compute UAA categories
            weight_df["UAA"] = weight_df.apply(lambda x: self.metadata[(x["UAA value"] >= self.metadata["ge UAA"])&
                                                                       (x["UAA value"] <  self.metadata["l UAA"])]["UAA"].unique()[0], axis=1)

            # Compute ES categories
            weight_df["ES"] = weight_df.apply(lambda x: self.metadata[(x["ES value"] >= self.metadata["ge ES"])&
                                                                      (x["ES value"] <  self.metadata["l ES"])]["ES"].unique()[0], axis=1)
            
            for es_cat in sorted(self.metadata["ES"].unique()):
                for uaa_cat in sorted(self.metadata["UAA"].unique()):

                    # Get number of farms in the data sample
                    n_sample = weight_df[(weight_df["ES"]==es_cat)&(weight_df["UAA"]==uaa_cat)].shape[0]

                    # Get real number of farms represented
                    n_real = self.metadata[(self.metadata["YEAR"]==self.YEAR)&(self.metadata["ES"]==es_cat)&(self.metadata["UAA"]==uaa_cat)]["value"].unique()

                    # If this sector is represented
                    if len(n_real) > 0 and n_sample > 0:

                        # Compute weights
                        weight = n_real[0] / n_sample + 1e-8
                        
                        weight_df.loc[weight_df[(weight_df["ES"]==es_cat)&(weight_df["UAA"]==uaa_cat)].index, "weight"] = weight

            microdata["A_TY_80_W"] = weight_df["weight"].fillna(0)
            
            print(f"Generared weights for use case {self.USE_CASE} using representativeness of year {self.YEAR}")
            print(f'Number total of farms: {microdata["A_TY_80_W"].sum()}\n\n\n')

            return microdata

        else:
            pass
