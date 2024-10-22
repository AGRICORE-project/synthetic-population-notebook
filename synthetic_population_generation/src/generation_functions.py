import numpy as np
import os
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from src.report_functions import *

from warnings import simplefilter
simplefilter(action="ignore", )#category=pd.errors.PerformanceWarning)

import math

def _generate_synthetic_sample(y_real, adj_mat_s, columns_spg, categoricals, VAR_TOL=1e-12):
    """
    Function to create synthetic data.

    Parameters
    ----------
    y_real: DataFrame
        origintal data to be replicated
    adj_mat_s: DataFrame
        adjacency matrix representing the structure of the Bayesian Network
    VAR_TOL: float
        minimum variance to do not consider the variable as constant

    Returns
    ----------
    y_synth: DataFrame
        synthetic population generated for the complete set of variables given
    """
    n_rows = y_real.shape[0]

    # Create a datafarme with the dimmensions and the column names of y_real
    y_synth = pd.DataFrame(columns=columns_spg, index=range(n_rows)).fillna(0)
    
    for i, newVar in enumerate(adj_mat_s.columns):
        
        # Get variables that affect newVar
        parent = adj_mat_s.index[adj_mat_s.iloc[:, i]==1].tolist()

        # Update is_categorical variable | in summary, is_categorical indicates if the variable is categorical
        is_categorical = True if newVar in categoricals else False
        
        # Remove variables whose variance is lower than VAR_TOL from parent list
        if len(parent) > 0:
            
            # From variables contained in parent, select those whose variance is lower than VAR_TOL
            selection = (y_real[parent].var() < VAR_TOL).to_frame().transpose()
            parent_constant = selection.loc[:, selection.any()].columns.tolist()
            
            # Remove the element a from parent given the condition
            for elem_ in parent_constant:
                parent.remove(elem_)

        # Call gen function
        y_synth[newVar] = _generate_column(y_synth[parent], y_real[newVar], y_real[parent], n_rows, is_categorical, newVar)

        # Fill nans
        y_synth.loc[:, newVar] = y_synth[newVar].fillna(0)
        #y_synth[newVar].fillna(0, inplace=True)

    return y_synth


def _knn(X_new, y_real, X_real, is_categorical, hyperparameters_neighbors=np.arange(1, 2)):
    """
    KNN algorithm to estimate y_synth given the parent variables from the synthetic sample that is being generated. 
    The algorithm is trained with the real data X_real (parent variables from real sample) and y_real (variable to
    be generated from the real sample). The algorithm purpose (classification or regression) is choosen given the
    value of is_categorical. The algorithm hyperparameters (n neighbours) are optimised by performing several trials 
    for the values given in hyperparameters_neighbors.

    Args
    ----------
    X_new: DataFrame
        dataframe with parent variables from synthetic sample
    y_real: DataFrame
        dataframe with the variable that will be generated from the real sample
    X_real: DataFrame
        dataframe with parent variables from the real sample
    is_categorical: bool
        if the variable that is being generated is categorical
    hyperparameters_neighbors: np.array
        array containing the values of k to be evaluated
    
    Returns
    ----------
    y_synth: np.array
        synthetic variable genenerated with KNN algorithm
    """
    
    # Get mean and standard deviation from X_real
    mean_, std_, _ = _get_mean_std_var(X_real)
    
    # Standarize X_real
    X_real = _standarization_function(X_real, mean_, std_)
    
    # Standarize X_new
    X_new = _standarization_function(X_new, mean_, std_)
    
    # Homogeinise data types
    y_real = y_real.astype(np.float32)
    
    # Use KNN classifier
    scores = []
    
    # Define a lambda function to select the KNN classifier required [classification / regression]
    _get_KNN_model = lambda is_categorical, n_neighbors: KNeighborsClassifier(n_neighbors=n_neighbors) if is_categorical \
        else KNeighborsRegressor(n_neighbors=n_neighbors)
    
    # Define a lambda function to select the appropiate metric to be used according to the KNN model selected
    _compute_score = lambda is_categorical, y_true, y_pred: np.mean(y_pred==y_true) if is_categorical \
        else -np.mean((y_true-y_pred)**2)
    
    # Explore all possible neighbour values
    for n_neighbors in hyperparameters_neighbors:
    
        # Get KNN model
        knn_model = _get_KNN_model(is_categorical, n_neighbors)

        # Fit model and predict with real data
        y_pred = knn_model.fit(X_real, y_real).predict(X_real)
        
        # Compute score and store result
        scores.append(_compute_score(is_categorical, y_real, y_pred))
        
    # Get optimal number of neighbours
    n_neighbors_opt = hyperparameters_neighbors[np.argmax(scores)]
    
    # Instantiate optimal KNN model
    opt_knn_model = _get_KNN_model(is_categorical, n_neighbors_opt)
    
    # Train model with real data. Predict with synthetic data
    predictions = opt_knn_model.fit(X_real, y_real).predict(X_new)

    # Apply some noise to the predictions
    std_noise = 1/20
    noise = np.random.normal(1, std_noise, predictions.shape[0])

    # Add noise to the predictions
    predictions = predictions*noise
    
    return predictions


def _probabilistic_knn(X_new, y_real, X_real, is_categorical):
    """
    KNN algorithm to estimate y_synth given the parent variables from the synthetic sample that is being generated. 
    The algorithm is trained with the real data X_real (parent variables from real sample) and y_real (variable to
    be generated from the real sample). The algorithm purpose (classification or regression) is choosen given the
    value of is_categorical. The algorithm hyperparameters (n neighbours) are optimised by performing several trials 
    for the values given in hyperparameters_neighbors.

    Args
    ----------
    X_new: DataFrame
        dataframe with parent variables from synthetic sample
    y_real: DataFrame
        dataframe with the variable that will be generated from the real sample
    X_real: DataFrame
        dataframe with parent variables from the real sample
    is_categorical: bool
        if the variable that is being generated is categorical
    hyperparameters_neighbors: np.array
        array containing the values of k to be evaluated
    
    Returns
    ----------
    y_synth: np.array
        synthetic variable genenerated with KNN algorithm
    """
    
    # If variable is categorical
    if is_categorical:
        # Create contingecy table for categories
        contingency_table = pd.crosstab(X_real, y_real)
        display(contingency_table)
        
        # Calculate probability of each category
        row_sums = contingency_table.sum(axis=1)

        # Normalize probabilities by parent value
        probabilities = contingency_table.apply(lambda x: x/row_sums, axis=0)
        
        # Create dataframe to store synthetic data
        y_synth = pd.DataFrame(X_new, columns=["Parent"])
        
        # Generate synthetic data
        y_synth["synthetic"] = y_synth.apply(lambda x: np.random.choice(probabilities.columns, 1, p=probabilities.loc[x["Parent"]])[0], axis=1)

        return y_synth["synthetic"].values
    
    else:
        # Standardise parents values
        min_ = X_real.min()
        max_ = X_real.max()
        
        X_real = X_real.apply(lambda x: (x - min_)/(max_-min_), axis=1)
        X_new = X_new.apply(lambda x: (x - min_)/(max_-min_), axis=1)

        # Create contingecy table for categories
        if len(X_real.shape)>1:
            xy = pd.concat([X_real.apply(lambda x: str(tuple(x)), axis=1), y_real], axis=1)
            contingency_table = pd.crosstab(xy[0], xy[y_real.name])

        else:
            contingency_table = pd.crosstab(X_real, y_real)
        
        # Calculate probability of each category
        row_sums = contingency_table.sum(axis=1)

        # Normalize probabilities by parent value
        probabilities = contingency_table.apply(lambda x: x/row_sums, axis=0)
        
        # Create dataframe to store synthetic data
        if isinstance(X_new, pd.Series):
            y_synth = X_new.to_frame()
        else:
            y_synth = X_new.copy(deep=True)

        # Convert index into numpy array
        index_ = np.array([eval(x) for x in probabilities.index])
        
        # Lambda function to find closest value in probabilities        
        closest_row = lambda x, index_: np.linalg.norm(x - index_, axis=1).argmin()

        # Compute closest probability row
        y_synth["closest"] = y_synth.apply(lambda x: closest_row(x.values, index_), axis=1)
        
        # Assign values P(x=0.0) = 1 .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
        # Zero-zero positions
        if "0.00" in probabilities.columns:
            probabilities_00 = probabilities[probabilities[0.00]==1].index

            # Convert variable values into list indexes
            index_list_00 = [probabilities.index.tolist().index(i_) for i_ in probabilities_00]

            # Assign P(x=0.0) = 1 to synthetic data
            y_synth["synth"] = y_synth.apply(lambda x: 0.0 if x["closest"] in index_list_00 else True, axis=1)

        else:
            y_synth["synth"] = True

        # .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
        # Assign P(x=0.0) < 1 to synthetic data
        y_synth.loc[y_synth["synth"]==True, "synth"] = y_synth.loc[y_synth["synth"]==True].apply(lambda x: np.random.choice(
            probabilities.columns, 
            1, 
            p=probabilities.iloc[x["closest"]])[0], axis=1)

        return y_synth["synth"].values


def _get_mean_std_var(x):
    
    # Compute the mean of x per column
    mean_ = x.mean()

    # Compute the standard deviation of x per column
    std_ = x.std()

    # Compute the variance of x per column
    var_ = x.var()

    return mean_, std_, var_


def _standarization_function(x, mean_, std_):
    
    x_std = (x - mean_) / std_

    return x_std


def _KDE_function(data, n_rows, newVar, muN=0, sigmaN=20):
    """
    Kernel Density Estimator function
    
    Parameters
    ----------
    data: array_like
        data to be fitted by the KDE
    n_rows: int
        number of rows of the given variable
    muN: float
        mean of the normal distribution to be generated
    sigmaN: float
        standard deviation of the distribution to be generated
    Returns
    ----------
    est: array_like
        data generated by the KDE
    """
    # Split data into zeros and non-zeros
    X_gen = data.copy(deep=True)
    X_nonzero = data[data!=0].copy(deep=True)
    
    # Check if all values must be positive
    positive = True if X_nonzero.min() >= 0 else False
    
    #.-.-.-.-.-.-.-.-..--.-..-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
    # Instantiate KDE model
    factor = 0.01343133231
    bw = factor*X_nonzero.mean()
    kde = KernelDensity(kernel="gaussian", bandwidth=bw)
    
    # Train KDE model
    kde.fit(X_nonzero.values.reshape(-1, 1))

    # Generate synthetic data
    X_gen_nonzero = kde.sample(X_nonzero.shape[0]).flatten()
    
    #.-.-.-.-.-.-.-.-..--.-..-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
    
    # Input synthetic-non zero values
    X_gen.loc[X_gen!=0] = X_gen_nonzero

    # Correct negative values if any
    if positive:
        X_gen.loc[X_gen<0] = np.abs(X_gen.loc[X_gen<0])

    return X_gen


def _generate_column(X_new, y_real, X_real, n_rows, is_categorical, newVar="", VAR_TOL=1e-12):
    """
    Function used to generate data using different methods
    
    Parameters
    ----------
    X_new: DataFrame
        dataframe with parent variables from synthetic sample
    y_real: DataFrame
        dataframe with the variable that will be generated from the real sample
    X_real: DataFrame
        dataframe with parent variables from the real sample
    is_categorical: bool
        if the variable that is being generated is categorical
    VAR_TOL: float
        value to determine if a variable should be copied directly from the original because of its variance is lower to VAR_TOL
    Returns
    ----------
    y_synth: DataFrame
        synthetic variable generated
    """
    #print(f"Generating variable {newVar}...")
    
    # If the variable to be generated has parent variables
    if X_real.shape[1] > 0:
        
        # Call knn algorithm to generate a new value from parents
        if is_categorical:
            y_synth = _knn(X_new, y_real, X_real, is_categorical)
        else:
            y_synth = _probabilistic_knn(X_new, y_real, X_real, is_categorical)

    # If the variable to be generated does not have parent variables
    else:
        # If variable to be generated is categorical
        if is_categorical:
            y_synth = y_real

        # If the variable to be generated is numerical
        else:
            # Get indexes where the variable is different from zero
            no_zero_indexes = np.where(y_real != 0)[0]

            # Initialise synthetic variable to zero
            y_synth = np.zeros(n_rows)

            # If the entire variable is zero -> assign zero
            if len(no_zero_indexes) == 0:
                pass

            # If only one instance is different from zero -> Generate a normal value with mean the non-zero value and std=1
            elif len(no_zero_indexes) == 1:
                y_synth[no_zero_indexes] = np.random.normal(y_real[no_zero_indexes], 1)
                
            # If more than one instance is different from zero
            else:
                
                # If variable is constant it has low variance, copy directly from original
                if np.var(y_real[no_zero_indexes]) < VAR_TOL:
                    y_synth[no_zero_indexes] = y_real[no_zero_indexes]
                    
                # If variable is not constant, it has a variance higher than var_tol, generte data using KDE
                else:
                    y_synth[no_zero_indexes] = _KDE_function(y_real[no_zero_indexes], len(no_zero_indexes), newVar)

                    if False:
                        BINS = 50
                        import matplotlib.pyplot as plt

                        #if newVar.endswith("cultivatedArea"):
                    
                        
                        plt.title(newVar)
                        
                        plt.hist(pd.DataFrame(y_real).fillna(0), bins=BINS, label="Original")
                        plt.hist(y_synth, bins=BINS, alpha=0.5, label="Synthetic")

                        plt.legend()
                        plt.show()

                        res_ = compute_statistics(pd.DataFrame(y_real, columns=[newVar]).fillna(0), 
                                                  pd.DataFrame(y_synth, columns=[newVar]), 
                                                  [], ".", "TRIAL", "REMOVE")
                        
                        display(res_)
                        #display(pd.concat([pd.DataFrame(y_real,  columns=["Original"]).fillna(0), 
                        #                   pd.DataFrame(y_synth, columns=["Synthetic"])], axis=1))
                        
                        print(y_real.shape, y_synth.shape)
                    
            if min(abs(y_real)) < 1e-12:
                ep = [i for i, val in enumerate(y_synth) if val < 0]
                if len(ep) > 0:
                    for i in ep:
                        y_synth[i] = y_real[i]
                        
    return y_synth


def _decode_data(y_synth, categoricals, encoder):
    """
    This function takes synthetic population with encoded data and reverses the encoding process of 
    categorical variables to restore the original values.

    Parameters
    ----------
    y_synth: pd.DataFrame
        synthetic population as DataFrame

    Returns
    ----------
    y_synth_decoded: pd.DataFrame
        synthetic population as DataFrame with categorical variables decoded
    """

    # Split categorical and numerical columns
    y_synth_numerical = y_synth[[c for c in y_synth.columns if c not in categoricals]]
    y_synth_categorical = y_synth[categoricals]

    # Replace encoded by decodes values in categoricals
    y_synth_categoricals_decoded = pd.DataFrame(encoder.inverse_transform(y_synth_categorical), columns=categoricals)
    
    # Join numerical columns and decoded categorical data
    y_synth_decoded = pd.concat([y_synth_numerical, y_synth_categoricals_decoded], axis=1)

    return y_synth_decoded