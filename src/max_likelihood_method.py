from utils import *
from experiment import setup_logging
import torch, math, scipy.stats as st
from torch.autograd.functional import hessian  # Fixed import - functorch is deprecated
import pandas as pd
from tqdm import tqdm
import os
from scipy.optimize import brentq
import numpy as np

def log_likelihood(network, rejection_set, p):
    # Calculate the log likelihood
    ll = 0
    eps = 1e-8
    for node in network.nodes():
        if 'p_value' in network.nodes[node]:
            reject = node in rejection_set
            # ancestors_index = torch.tensor(network.get_kinase_ancestors_index(node))
            parents_index = torch.tensor(network.get_kinase_parents_index(node))
            if len(parents_index) == 0:
                # Handle nodes with no kinase ancestors
                continue
            probability = torch.mean(p[parents_index])
            ll += reject * torch.log(probability+eps) + (1-reject) * torch.log(1-probability+eps)
    return ll

def nll(network, rejection_set, p):
    return -log_likelihood(network, rejection_set, p)




def main():
    logger = setup_logging('max_likelihood_method')
    logger.info("Starting max likelihood method")

    # network_path = 'data/interaction_mapping.csv'
    # member_path = 'data/dACC_Cohort_Information.xlsx'
    # residuals_path = 'data/all_residuals.csv'
    # category_path = 'data/category.csv'
    # case_df, control_df, network, p_values = load_data_and_network(logger, member_path, residuals_path, network_path, category_path, 'all')
    
    network = load_network_from_data(logger)
    rejection_path = 'results/exp_0.05_0.1_True_number_of_neighbor_STAR/rejection_set.csv'
    rejection_set = set(pd.read_csv(rejection_path)['node'].tolist())   
    
    # Get kinase names for results
    kinases = network.get_kinase()
    logger.info(f"Total number of kinases: {len(kinases)}")
    
    # Initialize the parameters 
    u = torch.nn.Parameter(torch.zeros(len(kinases)))
    
    # Optimize the parameters
    # 2) First use AdamW for coarse tuning
    # --------------------------
    logger.info("Coarse tuning with AdamW")
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
    logger.info("Fine tuning with LBFGS")
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

    # logger.info("=== MLE (first 10 dimensions example) ===")
    # for idx in range(min(10, len(p_hat))):
    #     logger.info(f"p[{idx}] = {p_hat[idx]:.4f}")

    # 4) Compute Profile Likelihood Confidence Intervals
    # --------------------------
    alpha = 0.2  # For 80% confidence intervals
    logger.info(f"Computing profile likelihood {100*(1-alpha):.0f}% confidence intervals...")
    
    n_params = len(kinases)
    lower_bounds = torch.zeros(n_params)
    upper_bounds = torch.zeros(n_params)
    
    # Compute profile likelihood intervals for each parameter
    for i in tqdm(range(n_params), desc="Computing profile likelihood intervals"):
        try:
            lower_bound, upper_bound = profile_likelihood_interval(
                network, rejection_set, u.detach(), i, alpha=alpha
            )
            lower_bounds[i] = lower_bound
            upper_bounds[i] = upper_bound
        except Exception as e:
            logger.warning(f"Failed to compute profile likelihood interval for parameter {i} ({kinases[i]}): {e}")
            # Fallback to point estimate with small bounds
            lower_bounds[i] = max(0.001, p_hat[i].item() - 0.1)
            upper_bounds[i] = min(0.999, p_hat[i].item() + 0.1)
    
    lower = lower_bounds
    upper = upper_bounds

    logger.info(f"\n=== Profile Likelihood {100*(1-alpha):.0f}% CI (first 10 dimensions example) ===")
    for idx in range(min(10, len(lower))):
        logger.info(f"{kinases[idx]}  CI{100*(1-alpha):.0f}% = [{lower[idx]:.4f}, {upper[idx]:.4f}]")

    # Save results to CSV
    # Create output directory if it doesn't exist
    output_dir = 'results/max_likelihood_method'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save optimization results for hyperparameter tuning (alpha selection)
    optimization_results = pd.DataFrame({
        'Name': kinases,
        'u_unconstrained': u.detach().cpu().numpy(),
        'p_hat_mle': p_hat.numpy(),
        'profile_lower_bound': lower.numpy(),
        'profile_upper_bound': upper.numpy()
    })
    
    optimization_path = os.path.join(output_dir, 'optimization_results.csv')
    optimization_results.to_csv(optimization_path, index=False)
    logger.info(f"Optimization results (u, p_hat, profile bounds) saved to {optimization_path}")
    
    # Create results DataFrame with current alpha
    results_df = pd.DataFrame({
        'Name': kinases,
        'Expectation': p_hat.numpy(),
        'Lower Bound': lower.numpy(),
        'Upper Bound': upper.numpy()
    })
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'kinase_results.csv')
    results_df.to_csv(output_path, index=False)
    logger.info(f"Kinase results with profile likelihood intervals (alpha={alpha}) saved to {output_path}")
    
    # Find top 30 kinases with highest lower bounds
    top_kinases = results_df.nlargest(30, 'Lower Bound')
    
    logger.info("Top 30 kinases with highest profile likelihood lower bounds:")
    for i, (_, row) in enumerate(top_kinases.iterrows(), 1):
        logger.info(f"{i:2d}. {row['Name']}: Lower Bound = {row['Lower Bound']:.4f}, "
                   f"Expectation = {row['Expectation']:.4f}, Upper Bound = {row['Upper Bound']:.4f}")
    
    logger.info(f"Maximum likelihood estimation with profile likelihood intervals completed. Results saved to {output_path}")
    logger.info(f"For different alpha values, rerun the script with modified alpha parameter")

if __name__ == "__main__":
    main()