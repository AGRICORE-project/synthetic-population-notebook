import pandas as pd
import os



class RepresentativenessComputer():
    def __init__(self, microdata, crop_codes, totals_variables_df):
        
        #self.product_mapping = crop_codes[["product_code", "Description", "CUSTOM GROUP (EN)"]]
        self.results = crop_codes[["product_code", "Description", "CUSTOM GROUP (EN)"]].copy(deep=True).rename(columns={"product_code": "fadn_code", "CUSTOM GROUP (EN)": "product_group"}).drop_duplicates().reset_index(drop=True)
        self.results["fadn_code_organic"] = self.results.apply(lambda x: f'{"ORG_" if x["product_group"].startswith("ORG") else ""}{x["fadn_code"]}', axis=1)

        # Make a selection of variables 
        used_variables = ["A_LO_40_N", "A_TY_90_ES", "A_TY_90_TF", "A_TY_80_W"]
        
        self.microdata = pd.concat([microdata.copy(deep=True), totals_variables_df], axis=1).dropna(axis=1, how="all")

        
        for var in ["B_UO_10_A", "B_UT_20_A", "B_US_30_A", "SE025", "SE135"]:
            if not var in self.microdata.columns:
                print(f"Missing: {var}")


    def compute_indicators(self, weight_var="A_TY_80_W"):
        """
        
        """

        self.results["n_appearances_abs"] = 0.0
        self.results["n_appearances_rel"] = 0.0
        self.results["total_area"] = 0.0
        self.results["production_quantity"] = 0.0
        self.results["sales_quantity"] = 0.0
        self.results["sales_value"] = 0.0
        
        
        for idx in self.results.index:

            crop = self.results.at[idx, "fadn_code_organic"]
            
            # 1. Crop counter
            count_ext = self.microdata.apply(lambda x: x[weight_var] if x[f"{crop}.cultivatedArea"]>0 else 0, axis=1).sum()
            self.results.at[idx, "n_appearances_abs"] = round(count_ext)
            
            # 2. Total area
            if f"{crop}.cultivatedArea" in self.microdata.columns:
                area_ext = self.microdata.apply(lambda x: x[weight_var]*x[f"{crop}.cultivatedArea"]/100, axis=1).sum()
                self.results.at[idx, "total_area"] = float(area_ext)
            
            # 3. Average area
            if f"{crop}.cultivatedArea" in self.microdata.columns:
                avg_area_ext = self.microdata.apply(lambda x: x[weight_var]*x[f"{crop}.cultivatedArea"]/100, axis=1).mean()
                self.results.at[idx, "average_area"] = float(avg_area_ext)
            
            # 4. Production quantity
            if f"{crop}.cropProduction" in self.microdata.columns:
                production_quantity_ext = self.microdata.apply(lambda x: x[weight_var]*x[f"{crop}.cropProduction"], axis=1).sum()
                self.results.at[idx, "production_quantity"] = float(production_quantity_ext)
            
            # 5. Sales quantity
            if f"{crop}.quantitySold" in self.microdata.columns:
                sales_quantity_ext = self.microdata.apply(lambda x: x[weight_var]*x[f"{crop}.quantitySold"], axis=1).sum()
                self.results.at[idx, "sales_quantity"] = float(sales_quantity_ext)
            
            # 6. Sales value
            if f"{crop}.valueSales" in self.microdata.columns:
                sales_value_ext = self.microdata.apply(lambda x: x[weight_var]*x[f"{crop}.valueSales"], axis=1).sum()
                self.results.at[idx, "sales_value"] = float(sales_value_ext)
            
            # 7. Share area
            if f"{crop}.cultivatedArea" in self.microdata.columns:
                if not "SE025" in self.microdata.columns:
                    if "agriculturalLandArea" in self.microdata.columns:
                        self.microdata["SE025"] = self.microdata["agriculturalLandArea"]
                    else:
                        self.microdata["SE025"] = self.microdata.apply(lambda x: ((x["B_UO_10_A"] + x["B_UT_20_A"] + x["B_US_30_A"])/100) if x["B_UO_10_A"] + x["B_UT_20_A"] + x["B_US_30_A"] > 0 else 0, axis=1).to_frame()
                    
                share_area = self.microdata.apply(lambda x: x[weight_var]*(x[f"{crop}.cultivatedArea"]/100) / x["SE025"] if x["SE025"] > 0 else 0, axis=1).to_frame()
                #share_area = share_area[share_area[0]>0].mean().item()
                share_area = share_area.mean().item()
                self.results.at[idx, "share_area"] = float(share_area)

            # 8. n crops rotation
            if f"{crop}.cultivatedArea" in self.microdata.columns:
                ta_variables = [cta for cta in self.microdata.columns if cta.endswith(".cultivatedArea")]
                selection = self.microdata[ta_variables + [weight_var]].copy(deep=True).fillna(0)
                selection = selection[selection[f"{crop}.cultivatedArea"]>0]
                selection_weight = selection[weight_var]
                selection = selection.drop(columns=weight_var)
                n_crops_mean = (selection>0).astype(int).sum(axis=1)#.mean()
                
                # Extrapolate average number of crops
                if selection_weight.sum() > 0:
                    n_crops_mean = (n_crops_mean.mul(selection_weight)).sum() / selection_weight.sum()
                    self.results.at[idx, "n_crops_combination"] = float(n_crops_mean)
                
        n_farms = round(self.microdata[weight_var].sum())

        self.results["n_appearances_rel"] = self.results.apply(lambda x: x["n_appearances_abs"]/n_farms, axis=1)
        
        return self.results
        