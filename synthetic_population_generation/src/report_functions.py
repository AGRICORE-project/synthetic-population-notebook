import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, cramervonmises_2samp, entropy
from scipy.spatial.distance import jensenshannon



def plot_categoricals(df1, df2, categoricals,REPORT_PATH, USE_CASE, YEAR):
    """
    Plot the distribution of the categorical variables in the original and synthetic data
    """

    if USE_CASE == "italy":
        variables_to_plot = [v_ for v_ in categoricals if (not v_.endswith("organicProductionType") and not v_ in ["farmCode", 
                                                                                                                    "weight_ra", 
                                                                                                                    "weight_reg", 
                                                                                                                    "regionLevel1", 
                                                                                                                    "regionLevel1Name", 
                                                                                                                    #"regionLevel3", 
                                                                                                                    #"regionLevel3Name", 
                                                                                                                    "holderSuccessors", 
                                                                                                                    #"holderFamilyMembers", 
                                                                                                                    "altitude", 
                                                                                                                    "holderSuccessorsAge"])]
        
    else:
        variables_to_plot = [v_ for v_ in categoricals if (not v_.endswith("organicProductionType") and not v_ in ["farmCode", 
                                                                                                                "weight_ra", 
                                                                                                                "weight_reg", 
                                                                                                                "regionLevel1", 
                                                                                                                "regionLevel1Name", 
                                                                                                                "regionLevel3", 
                                                                                                                "regionLevel3Name", 
                                                                                                                "holderSuccessors", 
                                                                                                                "holderFamilyMembers", 
                                                                                                                "holderSuccessorsAge"])]
    

    
    
    fig, axes = plt.subplots(nrows=len(variables_to_plot), ncols=1, figsize=(10, 3*len(variables_to_plot)))

    #plt.subplots_adjust(hspace=0.2)

    for i, var in enumerate(variables_to_plot):

        # Convert dfs into value counts
        df1_counts = df1[var].value_counts().to_dict()
        df2_counts = df2[var].value_counts().to_dict()

        df1_order = {}
        df2_order = {}

        for k in sorted(list(df1_counts.keys()) + list(df2_counts.keys())):
            if k in df1_counts.keys():
                df1_order[k] = df1_counts[k]
            else:
                df1_order[k] = 0

            if k in df2_counts.keys():
                df2_order[k] = df2_counts[k]
            else:
                df2_order[k] = 0
        
        # Create a unique index for the categories
        _index = {k: kk for k, kk in enumerate(df1_order.keys())}
        
        # Compute the width of the bars
        width = (list(_index.keys())[1]-list(_index.keys())[0])/3
        
        # Compute the offset displacement of the bars
        offset = width/2
        
        axes[i].bar([v-offset for v in _index.keys()], df1_order.values(), width=width, facecolor='blue', label='original', alpha=0.5)
        axes[i].bar([v+offset for v in _index.keys()], df2_order.values(), width=width, facecolor='red', label='synthetic', alpha=0.5)
        
        axes[i].legend()
        axes[i].set_title(var)

        axes[i].set_xticks([v for v in _index.keys()], [_index[k] for k in _index.keys()], rotation=90)

        if var=="holderGender":
            axes[i].set_xticks(range(0, 2), ["Male", "Female"], rotation=0)
        
    plt.tight_layout()

    plt.savefig(os.path.join(REPORT_PATH, f"categoricals_{USE_CASE}_{YEAR}.png"))
    
    plt.show()

    print(f"Categoricals plot saved in {REPORT_PATH}")



def compute_statistics(original, synthetic, categoricals, REPORT_PATH, USE_CASE, YEAR):

    # List containing the variables that will not be plotted
    ommited_variables = ["lat", "long", "rentBalanceOut", "rentBalanceIn", "weight_ra", "weight_reg", "farmCode", "variableCostsCrops", "milkVariableCosts", "rentBalance", "landValue", "A_CL_140_C", "yearNumber"]

    # List containing the variables to plot
    variables_to_plot = sorted([v for v in original.columns if (v not in categoricals + ommited_variables) and not np.sum([v.endswith(end_) for end_ in ommited_variables])>0])

    result_ = pd.DataFrame()

    for var in variables_to_plot:
        result_ = pd.concat([result_, compute_statistics_var(original[var], synthetic[var], var)], axis=0)
        
    # Save statistical report
    result_.to_csv(os.path.join(REPORT_PATH, f"statistics_{USE_CASE}_{YEAR}.csv"), index=True)

    return result_

def compute_statistics_var(original, synthetic, var):
    """
    Compute statistics of the synthetic data
    """
    
    # Kolmogorov-Smirnov Test
    ks_stat, ks_pvalue = ks_2samp(original, synthetic)
    
    # Cramer-von Mises Criterion
    cvm_stat = cramervonmises_2samp(original, synthetic).statistic
    cvm_pvalue = cramervonmises_2samp(original, synthetic).pvalue
    
    # Kullback-Leibler Divergence
    # Note: KL divergence requires non-zero probabilities; ensure no zero probabilities.
    def kl_divergence(p, q):
        p, q = np.array(p), np.array(q)
        return entropy(p + 1e-10, q + 1e-10)
    
    kl_div = kl_divergence(np.histogram(original, bins=30, density=True)[0], 
                           np.histogram(synthetic, bins=30, density=True)[0])
    
    # Jensen-Shannon Divergence
    def js_divergence(p, q):
        p, q = np.array(p), np.array(q)
        m = (p + q) / 2
        return (jensenshannon(p, m) + jensenshannon(q, m)) / 2
    
    js_div = js_divergence(np.histogram(original, bins=30, density=True)[0], 
                           np.histogram(synthetic, bins=30, density=True)[0])
    
    # Descriptive Statistics
    stats = {
        "min O":    [np.min(original)],
        "min S":   [np.min(synthetic)],
        "mean O":   [np.mean(original)],
        "mean S":  [np.mean(synthetic)],
        "max O":   [np.max(original)],
        "max S":   [np.max(synthetic)],
        "std O":    [np.std(original)], 
        "std S":   [np.std(synthetic)], 
        "ratio 0 O":  [np.sum(original==0)/len(original)],
        "ratio 0 S": [np.sum(synthetic==0)/len(synthetic)], 
        "KS p":  [ks_pvalue],
        "KS result":   ["Different" if ks_pvalue < 0.05 else "Similar"],
        "CVM p": [cvm_pvalue],
        "CVM result":  ["Different" if cvm_pvalue < 0.05 else "Similar"],
        "KL Div": [kl_div],
        "JS Div": [js_div],
    }
    
    return pd.DataFrame(stats, index=[var])
                

def batch(iterable, n=1):
    """
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


xlabel_dict = {
    "quantitySold":       "Quantity of Sold Production [tons]", 
    "valueSales":         "Value of Sales [€]", 
    "cropProduction":     "Value of total production [€]", 
    "irrigatedArea":      "Irrigated Area [ha]", 
    "cultivatedArea":     "Utilized Agricultural Area [ha]", 
    "quantityUsed":       "Quantity of Used Production [tons]", 
    "variableCostsCrops": "Variable Costs per produced unit [€/ton]",  
    "landValue":          "Land Value [€]", 
    "sellingPrice":       "Unit selling price [€/unit]", 
            
    "numberOfAnimals":              "Number of Animals [units]", 
    "numberOfAnimalsSold":          "Number of Animals Sold [units]", 
    "valueSoldAnimals":             "Value of Sold Animals [€]", 
    "numberAnimalsRearingBreading": "Number of Animals for Rearing/Breeding [units]", 
    "valueAnimalsRearingBreading":  "Value of Animals for Rearing/Breeding [€]", 
    "numberAnimalsForSlaughtering": "Number of Animals for Slaughtering [units]", 
    "valueSlaughteredAnimals":      "Value of Slaughtered Animals [€]", 
    
    "milkTotalProduction": "Number of tons of milk produced [tons]", 
    "milkProductionSold":  "Number of tons of milk sold [tons]", 
    "milkTotalSales":      "Value of milk sold [€]", 

    "milkVariableCosts":    "Variable Costs per\nproduced unit [€/ton]", 
    "dairyCows":            "Number of dairy cows [units]", 
    "variableCostsAnimals": "Cost per unit of product[€/ ton]", 
    
    "woolTotalProduction": "Wool Production Quantity [ton]", 
    "woolProductionSold":  "Wool Sales Quantity [€]", 
    
    "eggsTotalProduction": "Eggs Production Quantity [100·units]", 
    "eggsProductionSold":  "Eggs Sales Quantity [€]", 
    "eggsTotalSales":      "Eggs Sales Value [€]",

    "manureTotalSales": "Sales Value [€]", 
    
    "value": "Economic compensation [€]", 
    
    "agriculturalLandArea":                       "Total Agricultural Land [ha]", 
    "agriculturalLandValue":                      "Value of Agricultural Land [€]", 
    "agriculturalLandHectaresAdquisition":        "Acquired Agricultural Land [ha]", 
    "landImprovements":                           "Invesment in Land improvements [€]", 
    "forestLandArea":                             "Total Area of type Forest Land [ha]", 
    "forestLandValue":                            "Total value of Forest Land [€]", 
    "farmBuildingsValue":                         "Value of Buildings in the farm [€]", 
    "machineryAndEquipment":                      "Value of Machinery and Equipment[€]", 
    "intangibleAssetsTradable":                   "Value of tradable intangible assets [€]", 
    "intangibleAssetsNonTradable":                "Value of non-tradable intangible assets [€]", 
    "otherNonCurrentAssets":                      "Value of other non-current assets [€]", 
    "longAndMediumTermLoans":                     "Value of long and medium term loans [€]", 
    "totalCurrentAssets":                         "Total value of current assets [€]", 
    "farmNetIncome":                              "Farm Net Income [€]", 
    "grossFarmIncome":                            "Gross Farm Income [€]", 
    "subsidiesOnInvestments":                     "Value of subsidies on investments [€]", 
    "vatBalanceOnInvestments":                    "Balance of Taxes on Investments [€]", 
    "totalOutputCropsAndCropProduction":          "Value of Agricultural Production [€]", 
    "totalOutputLivestockAndLivestockProduction": "Value of Livestock Production [€]", 
    "otherOutputs":                               "Value of other outputs [€]", 
    "totalIntermediateConsumption":               "Value of intermediate consumption [€]", 
    "taxes":                                      "Value of Taxes [€]", 
    "vatBalanceExcludingInvestments":             "Balance of VAT excluding investments [€]", 
    "fixedAssets":                                "Value of Fixed Assets [€]", 
    "depreciation":                               "Yearly Depreciation [€]", 
    "totalExternalFactors":                       "Value of External Factors [€]", 
    "machinery":                                  "Value of Machinery [€]", 
    "rentPaid":                                   "Rent paid for land and buildings and rental changes [€]", 
    "rentBalance":                                "Balance (>0 received , <0 paid) of rent operations [€]", 
    "specificCropCosts":                          "Specific Crop costs [€/ton]", 
    "plantationsValue":                           "Seeds and plants [€]",

    "holderSuccessors":    "holderSuccessors", 
    "holderSuccessorsAge": "holderSuccessorsAge", 
    "holderFamilyMembers": "holderFamilyMembers", 
    "yearNumber":          "YEAR" # Note
    }


# Function to obtain the description and unit of the variable
#xlabel_limit = 35
#manage_xlabel = lambda x: x if len(x) < xlabel_limit else str(x[:[idx for idx, s in enumerate(x) if s == " "][-2]]) + "\n" + str(x[[idx for idx, s in enumerate(x) if s == " "][-2]+1:])
#set_xlabel = lambda x: manage_xlabel(xlabel_dict[x.split(".")[1]]) if "." in x else manage_xlabel(xlabel_dict[x])
set_xlabel = lambda x: xlabel_dict[x.split(".")[1]] if "." in x else xlabel_dict[x]


def make_plots(dfs, dfo, var_list, nrows, ncols, sheet_, REPORT_PATH, USE_CASE, YEAR, REPORT=True):
    """
    Make plots to compare original dataset with synthetic dataset variable pairs.
    
    Args:
        dfs (pd.DataFrame): Synthetic dataset.
        dfo (pd.DataFrame): Original dataset.
        var_list (list): List of variables to compare.
        nrows (int): Number of rows in the plot.
        ncols (int): Number of columns in the plot.
        sheet_ (int): sheet number in the report.
    """

    # 0. General plot settings
    with_density = False

    # Size settings
    width_cm  = 15
    height_cm = 24

    factor = 1.5
    width_inch  = width_cm*factor
    height_inch = height_cm*factor

    # Graphical parameters
    lim = 500
    ALPHA = 0.5
    BINS = 50
    fontsize_suptitle = 18
    fontsize_title = 15
    fontsize_label = 12

    sns.set(style="whitegrid")

    fig = plt.figure(figsize=(width_inch, height_inch), constrained_layout=True)

    subfigures = fig.subfigures(nrows=nrows, ncols=ncols, hspace=0.1)
    
    for sf_idx, var in enumerate(var_list):

        subfig_ = subfigures.flat[sf_idx]

        subfig_.suptitle(var, fontsize=fontsize_suptitle)
        axs = subfig_.subplots(nrows=1, ncols=3)

        # 1. Plot histograms
        if np.var(dfo[var]) == 0:
            sns.histplot(dfo[var], color="blue", bins=BINS, alpha=ALPHA, ax=axs[0], label="Original")
        else:
            sns.histplot(dfo[dfo[var]>0][var], color="blue", bins=BINS, alpha=ALPHA, ax=axs[0], label="Original")
        if np.var(dfs[var]) == 0:
            sns.histplot(dfs[var], color="red", bins=BINS, alpha=ALPHA, ax=axs[0], label="Synthetic")
        else:
            sns.histplot(dfs[dfs[var]>0][var], color="red", bins=BINS, alpha=ALPHA, ax=axs[0], label="Synthetic")
        
        axs[0].set_title("Histogram", fontsize=fontsize_title)
        #axs[0].set_xlabel(set_xlabel(var), fontsize=fontsize_label)
        axs[0].set_ylabel("Frequency >0", fontsize=fontsize_label)
        axs[0].legend(loc="best")

        if with_density:
            axs0_ = axs[0].twinx()
            # 2.1 Plot pdf (all)
            sns.kdeplot(dfo[dfo[var]>0][var], color="blue", linestyle=(0, (1, 5)), ax=axs0_)
            sns.kdeplot(dfs[dfs[var]>0][var], color="red",  linestyle=(0, (1, 5)), ax=axs0_)

            axs0_.set_ylabel("Density", fontsize=fontsize_label)


        # 2.1 Plot pdf
        if np.var(dfo[var]) > 0:
            sns.kdeplot(dfo[var], color="blue", ax=axs[1], alpha=ALPHA, label="Original")
        else:
            sns.histplot(dfo[var], ax=axs[1], stat="density", color="blue", bins=BINS, alpha=ALPHA, label="Original")

        if np.var(dfs[var]) > 0:
            sns.kdeplot(dfs[var], color="red",  ax=axs[1], alpha=ALPHA, label="Synthetic")
        else:
            sns.histplot(dfs[var], ax=axs[1], stat="density", color="red", bins=BINS, alpha=ALPHA, label="Synthetic")

        axs[1].set_title("PDF", fontsize=fontsize_title)
        #axs[1].set_xlabel(set_xlabel(var), fontsize=fontsize_label)
        axs[1].set_ylabel("Density", fontsize=fontsize_label)
        axs[1].legend(loc="best")

        # 3. Plot cdf
        if np.var(dfo[var]) > 0:
            sns.kdeplot(dfo[var], cumulative=True, color="blue", ax=axs[2], alpha=ALPHA, label="Original")
        else:
            sns.lineplot(pd.DataFrame({"x": [0, 1], "y": [0, 1]}), x="x", y="y", drawstyle="steps-pre", color="blue", ax=axs[2], alpha=ALPHA, label="Original")
        if np.var(dfs[var]) > 0:
            sns.kdeplot(dfs[var], cumulative=True, color="red",  ax=axs[2], alpha=ALPHA, label="Synthetic")
        else:
            sns.lineplot(pd.DataFrame({"x": [0, 1], "y": [0, 1]}), x="x", y="y", drawstyle="steps-pre", color="red", ax=axs[2], alpha=ALPHA, label="Synthetic")

        axs[2].set_title("CDF", fontsize=fontsize_title)
        #axs[2].set_xlabel(set_xlabel(var), fontsize=fontsize_label)
        axs[2].set_ylabel("Cumulative Density", fontsize=fontsize_label)
        axs[2].legend(loc="best")

        # Set common xlabel
        for sp in range(len(axs)):
            axs[sp].set_xlabel("")
        subfig_.text(0.5, -0.025, set_xlabel(var), ha='center', fontsize=fontsize_label+2)

    if REPORT:
        #plt.tight_layout()
        plt.savefig(os.path.join(REPORT_PATH, f"plot-variables_{USE_CASE}_{YEAR}_{sheet_}.png"),  bbox_inches="tight")#, pad_inches=0.1)
        print(f"Sheet {sheet_} saved in {REPORT_PATH}")
    else:
        plt.show()

    plt.close()


def compute_ratios(dfo, dfs, ratios_var):

    unit = " [ha]" if ratios_var=="cultivatedArea" else " [€]" if ratios_var=="cropProduction" else " [tons]" if ratios_var=="quantitySold" else ""

    r_df = pd.DataFrame()

    crops = [c.split(".")[0] for c in dfo.columns if c.endswith(ratios_var)]
    
    #r_df = pd.concat([r_df, pd.DataFrame({"  ": [" | " for _ in range(r_df.shape[0])]}, index=r_df.index)], axis=1)

    for crop in crops:

        r_df = pd.concat([r_df, pd.DataFrame({f"Original {ratios_var}":  dfo[f"{crop}.{ratios_var}"].sum().astype(int), 
                                              f"Synthetic {ratios_var}": dfs[f"{crop}.{ratios_var}"].sum().astype(int)}, index=[crop])], axis=0)
        
    r_df[f"Ratio {ratios_var}"] = r_df.apply(lambda x: x[f"Synthetic {ratios_var}"]/x[f"Original {ratios_var}"] if x[f"Original {ratios_var}"]>0 else 1, axis=1)
    
    r_df[f"Ratio {ratios_var}"] = r_df[f"Ratio {ratios_var}"].apply(lambda x: str(round(x, 3)))

    r_df = r_df.sort_values(f"Original {ratios_var}", ascending=False)
    
    r_df = r_df.rename(columns={f"Original {ratios_var}": f"Original {ratios_var}{unit}", 
                                f"Synthetic {ratios_var}": f"Synthetic {ratios_var}{unit}", })
    col_width = 150

    return r_df.style.set_table_styles([{'selector': 'th', 'props': [('min-width', f'{col_width}px'), ('max-width', f'{col_width}px')]},
                                        {'selector': 'td', 'props': [('min-width', f'{col_width}px'), ('max-width', f'{col_width}px')]}
        ])
