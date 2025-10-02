"""
contains the method to be used in the experiment.
currently contains the following methods:
BH
Bonferroni
STAR

return a list of node names
"""
from functools import reduce

import networkx as nx
from torch.autograd.functional import hessian  # Fixed import - functorch is deprecated
from collections import defaultdict
import scipy.stats as st
from scipy.optimize import brentq
from scipy.stats import chi2


import pandas as pd
import numpy as np
from scipy.stats import combine_pvalues, norm, chi2
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from vash import *
from utils import *
import torch
from tqdm import tqdm


def wald_interval(network, rejection_set, u, alpha):
    def nll_wrap(vec):                        # torch.autograd.functional.hessian needs function interface
        p_vec = torch.sigmoid(vec)
        return nll(network, rejection_set, p_vec)

    H = hessian(nll_wrap, u.detach())         # k×k Hessian
    try:
        cov = torch.linalg.inv(H)             # Covariance matrix
    except:
        cov = torch.pinverse(H)
        
    se  = torch.sqrt(torch.diagonal(cov))     # Standard error
    z      = st.norm.isf(alpha/2)             # Two-sided
    lower  = torch.sigmoid(u - z*se)
    upper  = torch.sigmoid(u + z*se)
    return lower, upper

def log_likelihood(network, rejection_set, p):
    # Calculate the log likelihood
    ll = 0
    eps = 1e-8
    for node in network.nodes():
        if 'p_value' in network.nodes[node]:
            reject = node in rejection_set
            parents_index = torch.tensor(network.get_kinase_parents_index(node))
            if len(parents_index) == 0:
                # Handle nodes with no kinase ancestors
                continue
            probability = torch.mean(p[parents_index])
            ll += reject * torch.log(probability+eps) + (1-reject) * torch.log(1-probability+eps)
    return ll

def nll(network, rejection_set, p):
    return -log_likelihood(network, rejection_set, p)

def profile_likelihood_interval(network, rejection_set, u_hat, param_idx, alpha, max_iter=50):
    """
    Compute profile likelihood confidence interval for parameter param_idx
    
    Args:
        network: The network object
        rejection_set: Set of rejected nodes
        u_hat: MLE estimates (unconstrained parameters)
        param_idx: Index of parameter to profile
        alpha: Significance level (e.g., 0.2 for 80% CI)
        max_iter: Maximum iterations for optimization
    
    Returns:
        (lower_bound, upper_bound): Profile likelihood confidence interval
    """
    n_params = len(u_hat)
    
    # Compute MLE likelihood value
    p_hat = torch.sigmoid(u_hat)
    ll_max = log_likelihood(network, rejection_set, p_hat).item()
    
    # Critical value from chi-square distribution
    critical_value = st.chi2.ppf(1 - alpha, df=1) / 2
    target_ll = ll_max - critical_value
    
    def profile_nll_at_fixed_p(p_fixed):
        """
        Profile negative log-likelihood when p[param_idx] is fixed at p_fixed
        Returns the minimized negative log-likelihood over all other parameters
        """
        # Create a copy of u_hat and fix the target parameter
        u_profile = u_hat.clone().detach()
        u_profile[param_idx] = torch.logit(torch.tensor(p_fixed, dtype=torch.float32))
        
        # Create mask for parameters to optimize (all except param_idx)
        optimize_mask = torch.ones(n_params, dtype=torch.bool)
        optimize_mask[param_idx] = False
        
        # Extract parameters to optimize
        u_free = torch.nn.Parameter(u_profile[optimize_mask].clone())
        
        # Optimize free parameters
        optimizer = torch.optim.LBFGS([u_free], lr=1.0, max_iter=max_iter,
                                     line_search_fn="strong_wolfe")
        
        def closure():
            optimizer.zero_grad()
            # Reconstruct full parameter vector
            u_full = u_profile.clone()
            u_full[optimize_mask] = u_free
            p_full = torch.sigmoid(u_full)
            loss = nll(network, rejection_set, p_full)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Get final likelihood
        with torch.no_grad():
            u_full = u_profile.clone()
            u_full[optimize_mask] = u_free
            p_full = torch.sigmoid(u_full)
            return -log_likelihood(network, rejection_set, p_full).item()
    
    def profile_diff(p_fixed):
        """
        Difference between profile likelihood and target likelihood
        """
        return -(profile_nll_at_fixed_p(p_fixed)) - target_ll
    
    # Find confidence interval bounds
    p_mle = torch.sigmoid(u_hat[param_idx]).item()
    
    # Search for lower bound
    try:
        # Start search from a point well below MLE
        p_low_start = max(0.001, p_mle - 0.3)
        if profile_diff(p_low_start) > 0:
            # Need to go lower
            p_search = p_low_start
            while p_search > 0.001 and profile_diff(p_search) > 0:
                p_search *= 0.5
            p_low_start = p_search
        
        lower_bound = brentq(profile_diff, p_low_start, p_mle, xtol=1e-4)
    except (ValueError, RuntimeError):
        # Fallback: use a very conservative lower bound
        lower_bound = 0.001
    
    # Search for upper bound  
    try:
        # Start search from a point well above MLE
        p_high_start = min(0.999, p_mle + 0.3)
        if profile_diff(p_high_start) > 0:
            # Need to go higher
            p_search = p_high_start
            while p_search < 0.999 and profile_diff(p_search) > 0:
                # print(p_search)
                # print(profile_diff(p_search))
                p_search = p_search + (0.9999 - p_search) * 0.5
            p_high_start = p_search
          
        upper_bound = brentq(profile_diff, p_mle, p_high_start, xtol=1e-4)
    except (ValueError, RuntimeError):
        # Fallback: use a very conservative upper bound
        upper_bound = 0.999
    
    return lower_bound, upper_bound

def get_kinase_ranking_new(network, rejection_set):
    kinases = network.get_kinase()
    
    # Initialize the parameters 
    u = torch.nn.Parameter(torch.zeros(len(kinases)))
    
    # Optimize the parameters
    # 2) First use AdamW for coarse tuning
    # --------------------------
    adam = torch.optim.AdamW([u], lr=0.02)
    for _ in tqdm(range(80)):                       # 80 steps are enough to pull parameters near convergence domain
        adam.zero_grad()
        p = torch.sigmoid(u)  # Move p calculation inside the loop
        loss = nll(network, rejection_set, p)
        loss.backward()
        adam.step()

    # --------------------------
    # 3) Then use LBFGS for fine tuning
    # --------------------------
    lbfgs = torch.optim.LBFGS([u], lr=1.0, history_size=20,
                            line_search_fn="strong_wolfe", max_iter=50)

    def closure():
        lbfgs.zero_grad()
        p = torch.sigmoid(u)  # Move p calculation inside closure
        loss = nll(network, rejection_set, p)
        loss.backward()
        return loss

    lbfgs.step(closure)
    u_hat = u.detach()
    n_params = len(kinases)
    p_hat = torch.sigmoid(u_hat)
    ll_max = log_likelihood(network, rejection_set, p_hat).item()

    def profile_nll_at_fixed_p(p_fixed, param_idx):
        """
        Profile negative log-likelihood when p[param_idx] is fixed at p_fixed
        Returns the minimized negative log-likelihood over all other parameters
        """
        # Create a copy of u_hat and fix the target parameter
        u_profile = u_hat.clone().detach()
        u_profile[param_idx] = torch.logit(torch.tensor(p_fixed, dtype=torch.float32))
        
        # Create mask for parameters to optimize (all except param_idx)
        optimize_mask = torch.ones(n_params, dtype=torch.bool)
        optimize_mask[param_idx] = False
        
        # Extract parameters to optimize
        u_free = torch.nn.Parameter(u_profile[optimize_mask].clone())
        
        # Optimize free parameters
        optimizer = torch.optim.LBFGS([u_free], lr=1.0, max_iter=100,
                                     line_search_fn="strong_wolfe")
        
        def closure():
            optimizer.zero_grad()
            # Reconstruct full parameter vector
            u_full = u_profile.clone()
            u_full[optimize_mask] = u_free
            p_full = torch.sigmoid(u_full)
            loss = nll(network, rejection_set, p_full)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Get final likelihood
        with torch.no_grad():
            u_full = u_profile.clone()
            u_full[optimize_mask] = u_free
            p_full = torch.sigmoid(u_full)
            return -log_likelihood(network, rejection_set, p_full).item()
    
    def profile_diff(p_fixed, param_idx):
        """
        Difference between profile likelihood and target likelihood
        """
        return (profile_nll_at_fixed_p(p_fixed, param_idx)) + ll_max
    
    # Find confidence interval bounds
    new_p_values = {}
    test_statistics = {}

    for i in range(len(kinases)):
        new_p_values[kinases[i]] = 1-chi2.cdf(2*profile_diff(0, i), df=1)
        test_statistics[kinases[i]] = profile_diff(0, i)
        # print(f"p_value for {kinases[i]}: {new_p_values[kinases[i]]}; profile_diff: {profile_diff(0, i)}")
    return new_p_values, test_statistics  
        

def get_kinase_ranking(network, rejection_set, CI=0.2):
    kinases = network.get_kinase()
    
    # Initialize the parameters 
    u = torch.nn.Parameter(torch.zeros(len(kinases)))
    
    # Optimize the parameters
    # 2) First use AdamW for coarse tuning
    # --------------------------
    adam = torch.optim.AdamW([u], lr=0.02)
    for _ in tqdm(range(80)):                       # 80 steps are enough to pull parameters near convergence domain
        adam.zero_grad()
        p = torch.sigmoid(u)  # Move p calculation inside the loop
        loss = nll(network, rejection_set, p)
        loss.backward()
        adam.step()

    # --------------------------
    # 3) Then use LBFGS for fine tuning
    # --------------------------
    lbfgs = torch.optim.LBFGS([u], lr=1.0, history_size=20,
                            line_search_fn="strong_wolfe", max_iter=100)

    def closure():
        lbfgs.zero_grad()
        p = torch.sigmoid(u)  # Move p calculation inside closure
        loss = nll(network, rejection_set, p)
        loss.backward()
        return loss

    lbfgs.step(closure)
    with torch.no_grad():
        p_hat = torch.sigmoid(u).cpu()        # Maximum likelihood estimation

    # 4) Compute Profile Likelihood Confidence Intervals
    # --------------------------
    n_params = len(kinases)
    lower_bounds, upper_bounds = wald_interval(network, rejection_set, u, CI) # by default, use wald interval
    # Compute profile likelihood intervals for each parameter
    epsilon = 0.05
    for i in tqdm(range(n_params), desc="Computing profile likelihood intervals"):
        try:
            if p_hat[i].item()<epsilon or p_hat[i].item()>1-epsilon: # use profile likelihood interval for boundary
                lower_bound, upper_bound = profile_likelihood_interval(
                    network, rejection_set, u.detach(), i, alpha=CI
                )
                lower_bounds[i] = lower_bound
                upper_bounds[i] = upper_bound

        except Exception as e:
            print(f"Failed to compute profile likelihood interval for parameter {i} ({kinases[i]}): {e}")
            # Fallback to point estimate with small bounds
            lower_bounds[i] = max(0.001, p_hat[i].item() - 0.1)
            upper_bounds[i] = min(0.999, p_hat[i].item() + 0.1)
    
    lower = lower_bounds
    upper = upper_bounds

    # Create results DataFrame with current alpha
    results_df = pd.DataFrame({
        'Name': kinases,
        'Expectation': p_hat.numpy(),
        'Lower Bound': lower.detach().cpu().numpy(),
        'Upper Bound': upper.detach().cpu().numpy()
    })
    # Find top 30 kinases with highest lower bounds
    top_kinases_df = results_df.nlargest(15, 'Lower Bound')
    top_kinases = top_kinases_df['Name'].tolist()
    return top_kinases, results_df

def get_pvalue_through_empirical_bayes(intensity_df, log_transform=True):
    '''
    get the p-value through empirical bayes
    '''

    dataset1 = intensity_df[intensity_df['group'] == 0]
    dataset2 = intensity_df[intensity_df['group'] == 1]
    merged_data = process_gene_pair_analysis(dataset1, dataset2, log_transform)
    vash_results = vash(sehat=merged_data['sehat'], betahat=merged_data['betahat'], \
                                       df=merged_data['degrees_of_freedom'])
    p_values = {}
    logFC = {}
    for i, molecule in enumerate(merged_data['molecules']):
        p_values[molecule] = vash_results['pvalue'][i]
        logFC[molecule] = merged_data['betahat'][i]
    return p_values, logFC

def BH_(p_values, alpha):
    '''Benjamini-Hochberg procedure, return all the significant molecules'''
    p_values = {k: v for k, v in sorted(p_values.items(), key=lambda item: item[1])}
    m = len(p_values)
    significant = []
    for i, (key, value) in enumerate(p_values.items()):
        if value <= alpha * (i+1) / m:
            p_values[key] = value
            significant.append(key)
        else:
            break
    return significant



def process_gene_pair_analysis(df1, df2, log_transform: bool = True) -> Dict[str, np.ndarray]:
    """
    Process a pair of datasets to get betahat and sehat for all shared genes.
    
    Args:
        dataset1_name, dataset2_name: Names of datasets to compare
        datasets: Dictionary of loaded datasets
        
    Returns:
        Dictionary containing gene names, betahat, and sehat arrays
    """
    
    # Get intensity column names for each dataset
    intensity_cols1 = get_intensity_columns(df1)
    intensity_cols2 = get_intensity_columns(df2)
    
    # Find shared molecules
    shared_molecules = find_shared_molecules(df1, df2)
    
    # Calculate statistics for each shared molecule
    betahat_list = []
    degrees_of_freedom_list = []
    sehat_list = []
    valid_molecules = []

    for molecule in shared_molecules:
        betahat, sehat, degrees_of_freedom = calculate_gene_statistics(df1, df2, molecule, intensity_cols1, intensity_cols2, log_transform)
        
        if not (np.isnan(betahat) or np.isnan(sehat)):
            betahat_list.append(betahat)
            sehat_list.append(sehat)
            degrees_of_freedom_list.append(degrees_of_freedom)
            valid_molecules.append(molecule)

    print(f"Successfully calculated statistics for {len(valid_molecules)} molecules")
    
    return {
        'molecules': np.array(valid_molecules),
        'betahat': np.array(betahat_list),
        'sehat': np.array(sehat_list),
        'degrees_of_freedom': np.array(degrees_of_freedom_list)
    }

def calculate_gene_statistics(df1: pd.DataFrame, df2: pd.DataFrame, 
                            gene_name: str, intensity_cols1: List[str], 
                            intensity_cols2: List[str], log_transform: bool = True) -> Tuple[float, float]:
    """
    Calculate betahat and sehat for a specific gene across two datasets.
    
    Args:
        df1, df2: DataFrames containing the data
        gene_name: Name of the gene to analyze
        intensity_cols1, intensity_cols2: Column names for intensity measurements
        
    Returns:
        Tuple of (betahat, sehat) where:
        - betahat: difference between means of the two datasets
        - sehat: standard error of the difference
    """
    # Get data for this gene from both datasets
    gene_data1 = df1[df1['name'] == gene_name]
    gene_data2 = df2[df2['name'] == gene_name]
    
    if len(gene_data1) == 0 or len(gene_data2) == 0:
        return np.nan, np.nan, np.nan
    
    # For genes with multiple rows, aggregate by taking mean?(what about missing value?)
    # intensities1 = np.array(gene_data1[intensity_cols1].values).mean(axis=0)
    # intensities2 = np.array(gene_data2[intensity_cols2].values).mean(axis=0)
    intensities1 = np.array(gene_data1[intensity_cols1].values)
    intensities2 = np.array(gene_data2[intensity_cols2].values)
    missing_intensities1 = intensities1!=0
    missing_intensities2 = intensities2!=0

    if missing_intensities1.sum() == 0 or missing_intensities2.sum() == 0 or missing_intensities1.sum() + missing_intensities2.sum() == 2:
        return np.nan, np.nan, np.nan
    
    if log_transform:
        intensities1 = np.log2(intensities1+1e-6)
        intensities2 = np.log2(intensities2+1e-6)

    # Calculate means
    mean1 = np.sum(intensities1*missing_intensities1)/np.sum(missing_intensities1)
    mean2 = np.sum(intensities2*missing_intensities2)/np.sum(missing_intensities2)
    
    # Calculate betahat (log fold change)
    betahat = mean2 - mean1
    
    # Calculate pooled standard error
    var1 = np.sum((intensities1-mean1)**2*missing_intensities1)/np.sum(missing_intensities1)
    var2 = np.sum((intensities2-mean2)**2*missing_intensities2)/np.sum(missing_intensities2)

    n1, n2 = np.sum(missing_intensities1), np.sum(missing_intensities2) 
    # TODO: according to kathryn and bernie, this shouldn't be the case in the duplicate cases; however, I should check the missingness first
    
    #Pooled variance
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    
    #Standard error of the difference between means
    sehat = np.sqrt(pooled_var * (1/n1 + 1/n2))
    
    #Add small constant to avoid zero standard errors
    sehat = max(sehat, 1e-6)
    return betahat, sehat, n1+n2-2

def get_intensity_columns(df):
    """
    Get intensity column names from a DataFrame.
    """
    columns = []
    for col in df.columns:
        if 'intensity' in col.lower():
            columns.append(col)
    return columns

def find_shared_molecules(df1, df2) -> List[str]:
    """
    Find genes that are present in both datasets.
    
    Args:
        df1, df2: DataFrames to compare
        
    Returns:
        List of shared gene names
    """
    molecules1 = set(df1['name'].dropna().unique())
    molecules2 = set(df2['name'].dropna().unique())
    shared_molecules = list(molecules1.intersection(molecules2))
    
    print(f"Found {len(shared_molecules)} shared molecules between the datasets")
    return shared_molecules
    
