import os
import pandas as pd
import numpy as np


class RegionLevelSpain():
    def __init__(self, base_path, use_case, year):
        
        self.USE_CASE = use_case
        self.YEAR = year

        self.BASE_PATH = base_path
        METADATA_PATH = os.path.join(self.BASE_PATH, "metadata")
        
        DISTRICTS_FILENAME = "regionLevel3.csv"
        
        self.DISTRICT_FILEPATH = os.path.join(METADATA_PATH, DISTRICTS_FILENAME)
    

    def _fill_regionLevel(self, sp):
        """
        Assign district value based on experimental likelihood.

        Parameters
        ----------
        sp: pd.DataFrame
            synthetic population

        Returns
        ----------
        sp: pd.DataFrame
            synthetic population with regionLevel and regionLevelName values filled
        """
        
        # Load districts data
        districts_data = pd.read_csv(self.DISTRICT_FILEPATH, encoding='latin-1', sep=";")

        # Remove unnecesary columns
        districts_data = districts_data.drop(columns=[c for c in [
            "Total Nacional", "Total", "SAU", "Cultivos, pastos y huertos", "Tipos de cultivos y pastos  II", "Tipo de cultivos III", "Tipo de cultivos IV", "Características básicas de la explotación", "Tamaño de las explotaciones según SAU (Ha.)", 
        ] if c in districts_data.columns])
        
        # Rename columns
        districts_data = districts_data.rename(columns={
            "Comunidades y Ciudades Autónomas": "regionLevel1",
            "Provincias": "regionLevel2", 
            "Comarcas": "regionLevel3",
            "Total.1": "Total",
        }).dropna()

        # Compose region names
        regionLevel1_dict = {
            "Andalucía": "ES61", 
            "Aragón": "ES24", 
            "Asturias, Principado de": "ES12", 
            "Balears, Illes": "ES53", 
            "Canarias": "ES70", 
            "Cantabria": "ES13", 
            "Castilla y León": "ES41", 
            "Castilla - La Mancha": "ES42", 
            "Cataluña": "ES51", 
            "Comunitat Valenciana": "ES52", 
            "Extremadura": "ES43", 
            "Galicia": "ES11", 
            "Madrid, Comunidad de": "ES30", 
            "Murcia, Región de": "ES62", 
            "Navarra, Comunidad Foral de": "ES22", 
            "País Vasco": "ES21", 
            "Rioja, La": "ES23", 
            "Ceuta": "ES63", 
            "Melilla": "ES64"
        }

        regionLevel2_dict = {'Almería': 'ES611', 'Cádiz': 'ES612', 'Córdoba': 'ES613', 'Granada': 'ES614', 'Huelva': 'ES615', 'Jaén': 'ES616', 'Málaga': 'ES617', 'Sevilla': 'ES618', 
                             'Huesca': 'ES241', 'Teruel': 'ES242', 'Zaragoza': 'ES243', 'Asturias': 'ES120', 'Balears, Illes': 'ES532', 'Palmas, Las': 'ES705', 'Santa Cruz de Tenerife': 'ES706', 
                             'Cantabria': 'ES130', 'Ávila': 'ES411', 'Burgos': 'ES412', 'León': 'ES413', 'Palencia': 'ES414', 'Salamanca': 'ES415', 'Segovia': 'ES416', 'Soria': 'ES417', 'Valladolid': 'ES418',
                             'Zamora': 'ES419', 'Albacete': 'ES421', 'Ciudad Real': 'ES422', 'Cuenca': 'ES423', 'Guadalajara': 'ES424', 'Toledo': 'ES425', 'Barcelona': 'ES511', 'Girona': 'ES512', 'Lleida': 'ES513',
                             'Tarragona': 'ES514', 'Alicante/Alacant': 'ES521', 'Castellón/Castelló': 'ES522', 'Valencia/València': 'ES523', 'Badajoz': 'ES431', 'Cáceres': 'ES432', 'Coruña, A': 'ES111', 'Lugo': 'ES112',
                             'Ourense': 'ES113', 'Pontevedra': 'ES114',  'Madrid': 'ES300', 'Murcia': 'ES620', 'Navarra': 'ES220', 'Araba/Álava': 'ES211', 'Bizkaia': 'ES212', 'Gipuzkoa': 'ES213', 'Rioja, La': 'ES230', 
                             'Ceuta': 'ES630', 'Melilla': 'ES640'}

        
        # Update regionLevel{}Name
        for level in [1, 2, 3]:
            districts_data[f"regionLevel{level}Name"] = districts_data.apply(lambda x: str(x[f"regionLevel{level}"][3 if level in [1, 2] else 5:]), axis=1)
            #districts_data[f"regionLevel{level}"] = districts_data.apply(lambda x: x[f"regionLevel{level}"][:2], axis=1)

        # Update regionLevel1 NUTS values
        districts_data[f"regionLevel1"] = districts_data.apply(lambda x: regionLevel1_dict[x["regionLevel1Name"]] if x["regionLevel1Name"] in regionLevel1_dict.keys() else "UNKNOWN", axis=1)

        # Update regionLevel2 NUTS values
        districts_data[f"regionLevel2"] = districts_data.apply(lambda x: regionLevel2_dict[x["regionLevel2Name"]] if x["regionLevel2Name"] in regionLevel2_dict.keys() else "UNKNOWN", axis=1)
        
        # Invert values by keys and vice-versa in regionLevel2_dict
        regionLevel1_dict = dict(zip(regionLevel1_dict.values(), regionLevel1_dict.keys()))
        regionLevel2_dict = dict(zip(regionLevel2_dict.values(), regionLevel2_dict.keys()))


        # Update regionLevel2 NUTS values
        districts_data[f"regionLevel3"] = districts_data.apply(lambda x: x["regionLevel2"] + x["regionLevel3"][2:4], axis=1)
            
        # Filter by NUTS2
        NUST2_population = list(sp["regionLevel1"].unique())
        districts_data = districts_data[districts_data["regionLevel1"].isin(NUST2_population)]
        
        # Convert totals value to numeric
        districts_data["Total"] = districts_data.apply(lambda x: int(x["Total"].replace(".", "")), axis=1)

        # Compute totals at different levels
        rl1_totals = districts_data[["regionLevel1", "Total"]].groupby(by="regionLevel1").sum().to_dict()["Total"]
        rl2_totals = districts_data[["regionLevel2", "Total"]].groupby(by="regionLevel2").sum().to_dict()["Total"]
        rl3_totals = districts_data[["regionLevel3", "Total"]].groupby(by="regionLevel3").sum().to_dict()["Total"]

        # Compute regionLevel2 probability
        districts_data["regionLevel1 proba"] = districts_data.apply(lambda x: rl1_totals[x["regionLevel1"]] / rl1_totals[x["regionLevel1"]], axis=1)
        districts_data["regionLevel2 proba"] = districts_data.apply(lambda x: rl2_totals[x["regionLevel2"]] / rl1_totals[x["regionLevel1"]], axis=1)
        districts_data["regionLevel3 proba"] = districts_data.apply(lambda x: rl3_totals[x["regionLevel3"]] / rl2_totals[x["regionLevel2"]], axis=1)

        # Create regionLevel3 mapping
        regionLevel3_dict = dict(zip(districts_data["regionLevel3Name"].tolist(), districts_data["regionLevel3"].tolist()))

        # display(districts_data)

        sp["regionLevel1Name"] = sp.apply(lambda x: regionLevel1_dict[x["regionLevel1"]], axis=1)
        
        # Assign regionLevel2 value if not available
        if sp["regionLevel2Name"].nunique()==1:

            if sp["regionLevel2"].nunique()==1:
                
                # Select regionLevel2 info
                districts_data_regionLevel2 = districts_data[["regionLevel2", "regionLevel2 proba"]].drop_duplicates()

                values = districts_data_regionLevel2["regionLevel2"].tolist()
                n_samples = sp.shape[0]
                probabilities = districts_data_regionLevel2["regionLevel2 proba"].tolist()
                
                # Generate regionLevel2 values
                synth_sample = np.random.choice(values, n_samples, p=probabilities)
                
                sp["regionLevel2"] = synth_sample
            sp["regionLevel2Name"] = sp.apply(lambda x: regionLevel2_dict[x["regionLevel2"]], axis=1)

        #.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
        # Assign regionLevel3 value if not available
        if sp["regionLevel3Name"].nunique()==1:
            for rl2 in sp["regionLevel2"].unique():
                
                sp_rl2 = sp[sp["regionLevel2"]==rl2]

                # Select reugionLevel3 info
                districts_data_regionLevel3 = districts_data[districts_data["regionLevel2"]==rl2][["regionLevel3Name", "regionLevel3 proba"]].drop_duplicates()

                values = districts_data_regionLevel3["regionLevel3Name"].tolist()
                n_samples = sp_rl2.shape[0]
                probabilities = districts_data_regionLevel3["regionLevel3 proba"].tolist()
            
                # Generate regionLevel3 values
                synth_sample = np.random.choice(values, n_samples, p=probabilities)
                
                # Assign values for regionLevel3Name
                sp.loc[sp_rl2.index, "regionLevel3Name"] = synth_sample
            
            # Assign
            sp["regionLevel3"] = sp.apply(lambda x: int(regionLevel3_dict[x["regionLevel3Name"]][2:]), axis=1)
            
        return sp



class RegionLevelGreece():
    def __init__(self, ):
        
        pass

    def _fill_regionLevel(self, sp):
        """
        Assign district value based on experimental likelihood.

        Parameters
        ----------
        sp: pd.DataFrame
            synthetic population

        Returns
        ----------
        sp: pd.DataFrame
            synthetic population with regionLevel and regionLevelName values filled
        """

        nuts_info = pd.DataFrame([
            ["NUTS2", "EL52",  "Kentriki Makedonia", 101337,], 
            ["NUTS3", "EL521", "Imathia",            13207, ], 
            ["NUTS3", "EL522", "Thesaloniki",        19342, ], 
            ["NUTS3", "EL523", "Kilkis",             10700, ], 
            ["NUTS3", "EL524", "Pella",              16926, ], 
            ["NUTS3", "EL525", "Pieria",             9141,  ], 
            ["NUTS3", "EL526", "Serres",             20193, ], 
            ["NUTS3", "EL527", "Chalkidiki",         11828, ], 
            ], columns=["NUTS", "NUTS code", "NUTS name", "Number of holdings"])

        #n_farms_real = sp["weight_ra"].astype(float).sum().astype(int)
        
        # Compute percentage belonging to each region
        nuts2_total = nuts_info[nuts_info["NUTS"]=="NUTS2"]["Number of holdings"].item()
        nuts_info["Number of holdings perc"] = nuts_info.apply(lambda x: x["Number of holdings"]/nuts2_total, axis=1)
        

        nuts2_info = nuts_info[nuts_info["NUTS"]=="NUTS2"]
        nuts2_link_dict = dict(zip(nuts2_info["NUTS code"].tolist(), nuts2_info["NUTS name"].tolist(), ))

        nuts3_info = nuts_info[nuts_info["NUTS"]=="NUTS3"]
        nuts3_link_dict = dict(zip(nuts3_info["NUTS code"].tolist(), nuts3_info["NUTS name"].tolist(), ))

        values = nuts3_info["NUTS code"].tolist()
        probabilities = nuts3_info["Number of holdings perc"].tolist()

        # Assignation regionLevel1
        sp["regionLevel1Name"] = sp.apply(lambda x: nuts2_link_dict[x["regionLevel1"]], axis=1)

        # Assignation regionLevel2
        sp["regionLevel2"] = np.random.choice(values, sp.shape[0], p=probabilities)
        sp["regionLevel2Name"] = sp.apply(lambda x: nuts3_link_dict[x["regionLevel2"]], axis=1)

        # Assignation regionLevel3
        # Divide data in 5
        n_rl3 = 5

        for rl2 in sp["regionLevel2"]:
            indexes = sp[sp["regionLevel2"]==rl2].index

            values = [f"{rl2}{i}" for i in range(1, 1+n_rl3)]
            probabilities = [1/n_rl3]*n_rl3

            sp.loc[indexes, "regionLevel3"] = np.random.choice(values, len(indexes), p=probabilities)

        sp["regionLevel3Name"] = sp.apply(lambda x: x["regionLevel3"], axis=1)

        # regionLevel must be an integer
        for rl in [1, 2, 3]:
            sp[f"regionLevel{rl}"] = sp.apply(lambda x: int(x[f"regionLevel{rl}"].replace("EL", "")), axis=1)

        return sp
    


class RegionLevelPoland():
    def __init__(self, ):
        
        pass

    def _fill_regionLevel(self, sp):
        """
        Assign district value based on experimental likelihood.

        Parameters
        ----------
        sp: pd.DataFrame
            synthetic population

        Returns
        ----------
        sp: pd.DataFrame
            synthetic population with regionLevel and regionLevelName values filled
        """

        nuts_info = pd.DataFrame([
            ["NUTS2", "PL81",  "Lubelskie", 1, ],

            ["NUTS3", "PL811", "Bialski",            0.25, ], 
            ["NUTS3", "PL812", "Chełmsko-zamojski",  0.25, ], 
            ["NUTS3", "PL814", "Lubelski",           0.25, ], 
            ["NUTS3", "PL815", "Puławski",           0.25, ], 
            
            ], columns=["NUTS", "NUTS code", "NUTS name", "Number of holdings perc"])

        display(nuts_info)

        
        #n_farms_real = sample_2014["A_TY_80_W"].sum().astype(int)
        n_farms_real = sp.shape[0]
        
        # Total number of farms
        print(f'Total Number of farms according to microdata in year 2014: {n_farms_real}')

        nuts2_info = nuts_info[nuts_info["NUTS"]=="NUTS2"]
        nuts2_link_dict = dict(zip(nuts2_info["NUTS code"].tolist(), nuts2_info["NUTS name"].tolist(), ))

        nuts3_info = nuts_info[nuts_info["NUTS"]=="NUTS3"]
        nuts3_link_dict = dict(zip(nuts3_info["NUTS code"].tolist(), nuts3_info["NUTS name"].tolist(), ))

        values = nuts3_info["NUTS code"].tolist()
        probabilities = nuts3_info["Number of holdings perc"].tolist()

        # Assignation regionLevel1
        sp["regionLevel1Name"] = sp.apply(lambda x: nuts2_link_dict[x["regionLevel1"]], axis=1)

        # Assignation regionLevel2
        sp["regionLevel2"] = np.random.choice(values, n_farms_real, p=probabilities)
        sp["regionLevel2Name"] = sp.apply(lambda x: nuts3_link_dict[x["regionLevel2"]], axis=1)

        # Assignation regionLevel3
        # Divide data in 5
        n_rl3 = 5

        print(".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")
        for rl2 in sp["regionLevel2"].unique():
            indexes = sp[sp["regionLevel2"]==rl2].index

            values = [f"{rl2}{i}" for i in range(1, 1+n_rl3)]
            probabilities = [1/n_rl3]*n_rl3

            sp.loc[indexes, "regionLevel3"] = np.random.choice(values, len(indexes), p=probabilities)
            

        sp["regionLevel3Name"] = sp.apply(lambda x: x["regionLevel3"], axis=1)

        # regionLevel must be an integer
        for rl in [1, 2, 3]:
            sp[f"regionLevel{rl}"] = sp.apply(lambda x: int(x[f"regionLevel{rl}"].replace("PL", "")), axis=1)

        
        return sp
    
