# vash.py
#
# This script is a Python translation of the R 'vash' package.
# It performs Variance Adaptive Shrinkage using Empirical Bayes methods.
#
# Dependencies:
# - numpy
# - scipy
# - statsmodels

import numpy as np
from scipy.special import gammaln, digamma, polygamma
from scipy.stats import t as t_dist
from scipy.optimize import minimize
from statsmodels.stats.multitest import fdrcorrection
import warnings
import matplotlib.pyplot as plt

from typing import List, Dict, Any, Optional, Tuple, Union

# A helper class to store the inverse-gamma mixture model parameters, similar to the R list 'g'
class Igmix:
    """A class to represent an Inverse-Gamma mixture model."""
    def __init__(self, pi: np.ndarray, alpha: np.ndarray, beta: np.ndarray, c: Optional[float] = None):
        self.pi = np.asarray(pi)
        self.alpha = np.asarray(alpha)
        self.beta = np.asarray(beta)
        self.c = c

    def ncomp(self) -> int:
        """Returns the number of mixture components."""
        return len(self.pi)

    def __repr__(self) -> str:
        return (f"Igmixed(pi={self.pi}, alpha={self.alpha}, "
                f"beta={self.beta}, c={self.c})")

# ==============================================================================
# Main VASH Function
# ==============================================================================

def vash(
    sehat: np.ndarray,
    df: Union[float, np.ndarray],
    betahat: Optional[np.ndarray] = None,
    randomstart: bool = False,
    singlecomp: bool = False,
    unimodal: str = "auto",
    prior: Optional[Union[str, list, np.ndarray]] = None,
    g: Optional[Igmix] = None,
    estpriormode: bool = True,
    priormode: Optional[float] = None,
    scale: Union[float, np.ndarray] = 1.0,
    maxiter: int = 5000
) -> Dict[str, Any]:
    """
    Main Variance Adaptive SHrinkage function.

    Takes vectors of standard errors (sehat), and applies shrinkage to them,
    using Empirical Bayes methods, to compute shrunk estimates for variances.

    Args:
        sehat: A vector of observed standard errors.
        df: Scalar or vector, appropriate degree of freedom for (chi-square) distribution of sehat.
        betahat: A vector of estimates (optional).
        randomstart: If True, initialize EM randomly.
        singlecomp: If True, use a single inverse-gamma distribution as the prior.
        unimodal: Unimodal constraint on prior ("variance", "precision", or "auto").
        prior: Dirichlet prior on mixture proportions. Defaults to "uniform".
        g: The prior distribution for variances (usually estimated from the data).
        estpriormode: If True, estimate the mode of the unimodal prior.
        priormode: Specified prior mode (only used if estpriormode=False).
        scale: A scalar or vector for scaling sehat.
        maxiter: Maximum number of EM iterations.

    Returns:
        A dictionary containing the fitted model, posterior estimates, and other results.
    """
    # 1. Handling Input Parameters
    if unimodal not in ["auto", "variance", "precision"]:
        raise ValueError("Error: invalid type of unimodal.")
    if betahat is not None and len(betahat) != len(sehat):
        raise ValueError("Error: sehat and betahat must have same lengths.")
    if g is not None and not isinstance(g, Igmix):
        raise ValueError("Error: invalid type of g.")
    
    # Updated validation for df - allow scalar or vector
    df = np.asarray(df, dtype=float)
    if df.ndim > 1:
        raise ValueError("Error: df must be a scalar or 1D array.")
    if df.ndim == 1 and len(df) != len(sehat):
        raise ValueError("Error: if df is a vector, it must have the same length as sehat.")
    if np.any(df <= 0):
        raise ValueError("Error: df must be positive.")

    # Prepare data
    sehat = np.asarray(sehat, dtype=float)
    original_sehat = sehat.copy()
    original_df = df.copy()
    sehat[np.isinf(sehat)] = np.nan
    sehat = sehat / scale
    if np.any(sehat[~np.isnan(sehat)] < 0):
        raise ValueError("Error: sehat/scale must be non-negative.")

    complete_obs = ~np.isnan(sehat)
    n_total = len(sehat)
    n_complete = np.sum(complete_obs)

    if n_complete == 0:
        raise ValueError("Error: all input values are missing.")

    # Add pseudocount to near-zero standard errors
    sehat_complete = sehat[complete_obs]
    zero_mask = sehat_complete == 0
    if np.any(zero_mask):
        min_positive = np.min(sehat_complete[sehat_complete > 0]) if np.any(sehat_complete > 0) else 1e-6
        sehat_complete[zero_mask] = min(min_positive, 1e-6)
    sehat[complete_obs] = sehat_complete

    # Subset df for complete observations
    if df.ndim == 0:  # scalar
        df_complete = df
    else:  # vector
        df_complete = df[complete_obs]

    # 2. Fitting the mixture prior
    opt_unimodal = None
    if unimodal == 'auto' and not singlecomp:
        pifit_prec = est_prior(sehat, df_complete, betahat, randomstart, singlecomp, 'precision',
                               prior, g, maxiter, estpriormode, priormode, complete_obs)
        pifit_var = est_prior(sehat, df_complete, betahat, randomstart, singlecomp, 'variance',
                              prior, g, maxiter, estpriormode, priormode, complete_obs)

        if pifit_prec['loglik'] >= pifit_var['loglik']:
            mix_fit = pifit_prec
            opt_unimodal = 'precision'
        else:
            mix_fit = pifit_var
            opt_unimodal = 'variance'
    else:
        current_unimodal = 'variance' if unimodal == 'auto' else unimodal
        mix_fit = est_prior(sehat, df_complete, betahat, randomstart, singlecomp, current_unimodal,
                            prior, g, maxiter, estpriormode, priormode, complete_obs)

    # 3. Posterior inference
    g_fit = mix_fit['g']
    post_se = post_igmix(g_fit, sehat[complete_obs], df_complete)

    # Initialize posterior matrices with prior values
    postpi_se = np.tile(g_fit.pi, (n_total, 1))
    post_shape_se = np.tile(g_fit.alpha, (n_total, 1))
    post_rate_se = np.tile(g_fit.beta, (n_total, 1))
    post_mean_se = np.full(n_total, np.nan) # Will be filled for complete obs

    # Compute posterior probabilities for complete observations
    data = {'s': sehat[complete_obs], 'v': df_complete}
    postpi_se[complete_obs, :] = comp_postprob(g_fit, data)

    # Update posterior shape and rate for complete observations
    post_shape_se[complete_obs, :] = post_se['alpha']
    post_rate_se[complete_obs, :] = post_se['beta']

    # Compute posterior mean of standard deviation
    # E[s] = E[1/sqrt(precision)] approx 1/sqrt(E[precision])
    # E[precision] = sum(pi_k * E[precision_k]) = sum(pi_k * alpha_k / beta_k)
    posterior_precision = np.sum(postpi_se * (post_shape_se / post_rate_se), axis=1)
    post_mean_se = 1 / np.sqrt(posterior_precision)
    post_mean_se[posterior_precision == 0] = np.inf

    # Compute p-values and q-values if betahat is provided
    pvalue, qvalue = None, None
    if betahat is not None:
        if len(betahat) == n_total:
            # Posterior mean of variance for t-test is E[1/precision] = beta/alpha
            moderated_se = np.sqrt(post_rate_se / post_shape_se) * scale
            # For mod_t_test, we need df to be broadcasted to (n, k) shape
            if original_df.ndim == 0:  # scalar
                df_for_ttest = np.full((n_total, post_shape_se.shape[1]), original_df * 2)
            else:  # vector
                df_for_ttest = np.tile((original_df * 2)[:, np.newaxis], (1, post_shape_se.shape[1]))
            pvalue = mod_t_test(betahat, moderated_se, postpi_se, df_for_ttest)
            pvalue_one_side = mod_t_test_one_side(betahat, moderated_se, postpi_se, df_for_ttest)
            
            # Filter out NaNs for q-value calculation
            valid_pvals = pvalue[~np.isnan(pvalue)]
            if len(valid_pvals) > 0:
                qvalue = np.full(n_total, np.nan)
                _, qvals = fdrcorrection(valid_pvals, alpha=0.1, method='indep')
                qvalue[~np.isnan(pvalue)] = qvals
        else:
            warnings.warn("betahat has different length than sehat, cannot compute moderated t-tests")

    result = {
        'fitted_g': g_fit,
        'sd_post': post_mean_se,
        'PosteriorPi': postpi_se,
        'PosteriorShape': post_shape_se,
        'PosteriorRate': post_rate_se,
        'pvalue': pvalue,
        'pvalue_oneside': pvalue_one_side,
        'qvalue': qvalue,
        'unimodal': unimodal,
        'opt_unimodal': opt_unimodal,
        'fit': mix_fit,
        'data': {'sehat': original_sehat, 'betahat': betahat, 'df': original_df}
    }
    return result

# ==============================================================================
# Prior Estimation Functions
# ==============================================================================

def est_prior(sehat, df, betahat, randomstart, singlecomp, unimodal,
              prior, g, maxiter, estpriormode, priormode, complete_obs):
    """Fits the mixture inverse-gamma prior of variance."""
    sehat_complete = sehat[complete_obs]

    # Set up initial parameters
    if g is not None:
        maxiter = 1  # If g is specified, don't iterate
        estpriormode = False
        l = g.ncomp()
        prior_val = np.ones(l)
    else:
        mm = est_singlecomp_mm(sehat_complete, df)
        mm_a = max(mm['a'], 1e-5)
        mm_b = max(mm['b'], 1e-5)
        
        if singlecomp:
            alpha = np.array([mm_a])
        else:
            if mm_a > 1:
                alpha = 1 + ((64 / mm_a)**(1/6))**np.arange(-3, 7) * (mm_a - 1)
            else:
                alpha = mm_a * 2**np.arange(14)
        
        if unimodal == 'precision':
            alpha = np.unique(np.maximum(alpha, 1 + 1e-5))
            beta = mm_b / (mm_a - 1) * (alpha - 1) if mm_a > 1 else mm_b * (alpha - 1)
        elif unimodal == 'variance':
            beta = mm_b / (mm_a + 1) * (alpha + 1)

        if not estpriormode and priormode is not None:
            if unimodal == 'precision':
                beta = priormode * (alpha - 1)
            elif unimodal == 'variance':
                beta = priormode * (alpha + 1)
        elif estpriormode and priormode is not None:
            warnings.warn('Flag estpriormode=True, vash will still estimate the prior mode.')
        
        l = len(alpha)

    # Set up prior on mixture proportions
    if prior is None or prior == "uniform":
        prior_val = np.ones(l)
    elif isinstance(prior, str) and prior == "nullbiased":
        prior_val = np.ones(l) / (l - 1) if l > 1 else np.ones(l)
        prior_val[0] = 1
    elif isinstance(prior, (list, np.ndarray)):
        prior_val = np.asarray(prior)
    else:
        raise ValueError("Error: invalid prior specification")
        
    if len(prior_val) != l:
        raise ValueError("Error: prior must have the same length as the number of components.")

    # Initialize mixture proportions
    if randomstart:
        pi_se = np.random.gamma(1, 1, l)
    else:
        pi_se = np.ones(l) / l
    pi_se /= np.sum(pi_se)
    
    if g is None:
        g = Igmix(pi_se, alpha, beta)

    # Fit the mixture prior
    mix_fit = est_mixprop_mode(sehat_complete, g, prior_val, df, unimodal,
                               singlecomp, estpriormode, maxiter)
    return mix_fit

def est_mixprop_mode(sehat, g, prior, df, unimodal, singlecomp, estpriormode, maxiter=5000):
    """Estimates mixture proportions and mode of the unimodal inverse-gamma mixture variance prior."""
    pi_init = g.pi
    n = len(sehat)
    tol = min(0.1 / n, 1e-4)
    
    if unimodal == 'variance':
        c_init = g.beta[0] / (g.alpha[0] + 1)
    elif unimodal == 'precision':
        c_init = g.beta[0] / (g.alpha[0] - 1) if g.alpha[0] > 1 else 1.0
    c_init = max(c_init, 1e-5)

    em_fit = ig_mix_em(sehat, df, c_init, g.alpha, pi_init, prior, unimodal,
                       singlecomp, estpriormode, tol, maxiter)
    
    g.pi = em_fit['pihat']
    g.c = em_fit['chat']
    
    if singlecomp:
        g.alpha = np.array([em_fit['alphahat']])
        
    if unimodal == 'variance':
        g.beta = g.c * (g.alpha + 1)
    elif unimodal == 'precision':
        g.beta = g.c * (g.alpha - 1)

    loglik = loglike_vash(sehat, df, g.pi, g.alpha, g.beta)

    return {'loglik': loglik, 'converged': em_fit['converged'], 'g': g, 'niter': em_fit['niter']}

# ==============================================================================
# EM Algorithm and Optimization
# ==============================================================================

def ig_mix_em(sehat, v, c_init, alpha_vec, pi_init, prior, unimodal, singlecomp, estpriormode, tol, maxiter):
    """
    EM algorithm to fit the mixture inverse-gamma prior.
    This replaces the R SQUAREM implementation with a simpler fixed-point iteration.
    """
    q = len(pi_init)
    n = len(sehat)

    if unimodal == 'variance':
        modalpha_vec = alpha_vec + 1
    else:  # precision
        modalpha_vec = alpha_vec - 1

    converged = False
    niter = 0
    
    if not singlecomp:
        A = get_A(n, q, v, alpha_vec, modalpha_vec, sehat)
        
        if estpriormode:
            # Iterate updating both pi and c
            params = np.concatenate(([np.log(c_init)], pi_init))
            for i in range(maxiter):
                niter += 1
                params_old = params.copy()
                params = fixpoint_se(params, A, n, q, alpha_vec, modalpha_vec, v, sehat, prior)
                if np.sum(np.abs(params - params_old)) < tol:
                    converged = True
                    break
            logc, pihat = params[0], params[1:]
            chat = np.exp(logc)
            alphahat = None
        else:
            # Iterate updating only pi
            pi = pi_init.copy()
            for i in range(maxiter):
                niter += 1
                pi_old = pi.copy()
                pi = fixpoint_pi(pi, A, n, q, alpha_vec, modalpha_vec, v, sehat, prior, c_init)
                if np.sum(np.abs(pi - pi_old)) < tol:
                    converged = True
                    break
            chat, pihat = c_init, pi
            alphahat = None
    else: # singlecomp
        params_init = np.array([np.log(c_init), np.log(alpha_vec[0])])
        bounds = [(None, None), (-3, np.log(100))]
        
        res = minimize(loglike_se_ac, params_init, args=(n, 1, v, sehat, np.array([1.0]), unimodal),
                       jac=gradloglike_se_ac, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'pgtol': tol})
        
        logc, logalpha = res.x
        chat = np.exp(logc)
        pihat = np.array([1.0])
        alphahat = np.exp(logalpha)
        niter = res.nit
        converged = res.success

    return {'chat': chat, 'pihat': pihat, 'niter': niter, 'converged': converged, 'alphahat': alphahat}


def fixpoint_se(params, A, n, k, alpha_vec, modalpha_vec, v, sehat, prior):
    """Fixed-point update for both pi and c."""
    logc, pi = params[0], params[1:k+1]
    
    mm = post_pi_vash(A, n, k, v, sehat, alpha_vec, modalpha_vec, np.exp(logc), pi)
    classprob = mm / np.sum(mm, axis=1, keepdims=True)
    
    newpi = np.sum(classprob, axis=0) + prior - 1
    newpi = np.maximum(newpi, 1e-5)
    newpi /= np.sum(newpi)
    
    # Ensure newpi is 1D
    newpi = np.atleast_1d(newpi).flatten()
    
    res = minimize(loglike_se, logc, args=(n, k, alpha_vec, modalpha_vec, v, sehat, newpi),
                   jac=gradloglike_se, method='BFGS')
    newlogc = np.atleast_1d(res.x).flatten()[0]
    
    return np.concatenate(([newlogc], newpi))

def fixpoint_pi(pi, A, n, k, alpha_vec, modalpha_vec, v, sehat, prior, c):
    """Fixed-point update for pi only."""
    mm = post_pi_vash(A, n, k, v, sehat, alpha_vec, modalpha_vec, c, pi)
    classprob = mm / np.sum(mm, axis=1, keepdims=True)
    
    newpi = np.sum(classprob, axis=0) + prior - 1
    newpi = np.maximum(newpi, 1e-5)
    newpi /= np.sum(newpi)
    
    # Ensure newpi is 1D
    newpi = np.atleast_1d(newpi).flatten()
    
    return newpi

# ==============================================================================
# Likelihood and Gradient Functions
# ==============================================================================

def get_A(n, k, v, alpha_vec, modalpha_vec, sehat):
    """Pre-computes a part of the posterior calculation."""
    # Handle vector v (df) with proper broadcasting
    if np.ndim(v) == 0:  # scalar
        return (v/2 * np.log(v/2) - gammaln(v/2) +
                (v/2 - 1) * (2 * np.log(sehat))[:, np.newaxis] +
                (alpha_vec * np.log(modalpha_vec) - gammaln(alpha_vec) +
                 gammaln(alpha_vec + v/2))[np.newaxis, :])
    else:  # vector
        v = np.asarray(v)
        v_half = v / 2
        return ((v_half * np.log(v_half) - gammaln(v_half) +
                (v_half - 1) * (2 * np.log(sehat)))[:, np.newaxis] +
                (alpha_vec * np.log(modalpha_vec) - gammaln(alpha_vec) +
                 gammaln(alpha_vec[np.newaxis, :] + v_half[:, np.newaxis])))

def post_pi_vash(A, n, k, v, sehat, alpha_vec, modalpha_vec, c, pi):
    """Computes unnormalized posterior probabilities."""
    # Handle vector v (df) with proper broadcasting
    if np.ndim(v) == 0:  # scalar
        log_pi_mat = (A + (alpha_vec * np.log(c))[np.newaxis, :] -
                      (alpha_vec + v/2)[np.newaxis, :] *
                      np.log((c * modalpha_vec)[np.newaxis, :] + (v/2 * sehat**2)[:, np.newaxis]))
    else:  # vector
        v = np.asarray(v)
        v_half = v / 2
        log_pi_mat = (A + (alpha_vec * np.log(c))[np.newaxis, :] -
                      (alpha_vec[np.newaxis, :] + v_half[:, np.newaxis]) *
                      np.log((c * modalpha_vec)[np.newaxis, :] + (v_half[:, np.newaxis] * sehat[:, np.newaxis]**2)))
    
    log_pi_mat += np.log(pi)[np.newaxis, :]
    
    # Subtract max for numerical stability before exponentiating
    max_log = np.max(log_pi_mat, axis=1, keepdims=True)
    post_pi_mat = np.exp(log_pi_mat - max_log)
    return post_pi_mat
    
def loglike_vash(sehat, df, pi, alpha, beta):
    """Computes the log-likelihood of the variance model."""
    k = len(pi)
    n = len(sehat)
    
    # Handle vector df with proper broadcasting
    if np.ndim(df) == 0:  # scalar
        df_terms = (df/2 * np.log(df/2) - gammaln(df/2) +
                   (df/2 - 1) * (2 * np.log(sehat))[:, np.newaxis])
        alpha_df_terms = (alpha * np.log(beta) - gammaln(alpha) +
                         gammaln(alpha + df/2))[np.newaxis, :]
        denom_terms = ((alpha + df/2)[np.newaxis, :] *
                      np.log((beta)[np.newaxis, :] + (df/2 * sehat**2)[:, np.newaxis]))
    else:  # vector
        df = np.asarray(df)
        df_half = df / 2
        # Ensure all arrays are broadcast to (n, k) shape
        df_terms = ((df_half * np.log(df_half) - gammaln(df_half) +
                    (df_half - 1) * (2 * np.log(sehat)))[:, np.newaxis] *
                   np.ones((1, k)))  # Broadcast to (n, k)
        alpha_df_terms = ((alpha * np.log(beta) - gammaln(alpha))[np.newaxis, :] +
                         gammaln(alpha[np.newaxis, :] + df_half[:, np.newaxis]))
        denom_terms = ((alpha[np.newaxis, :] + df_half[:, np.newaxis]) *
                      np.log(beta[np.newaxis, :] + (df_half[:, np.newaxis] * sehat[:, np.newaxis]**2)))
    
    log_lik_matrix = (np.log(pi)[np.newaxis, :] + df_terms + alpha_df_terms - denom_terms)
    
    max_log = np.max(log_lik_matrix, axis=1, keepdims=True)
    lik_matrix = np.exp(log_lik_matrix - max_log)
    logl = np.sum(np.log(np.sum(lik_matrix, axis=1)) + max_log.flatten())
    return logl

def loglike_se(logc, n, k, alpha_vec, modalpha_vec, v, sehat, pi):
    """Negative log-likelihood w.r.t. log(c)."""
    c = np.exp(logc)
    beta = c * modalpha_vec
    return -loglike_vash(sehat, v, pi, alpha_vec, beta)

def gradloglike_se(logc, n, k, alpha_vec, modalpha_vec, v, sehat, pi):
    """Gradient of negative log-likelihood w.r.t. log(c)."""
    c = np.exp(logc)
    beta = c * modalpha_vec
    
    # Handle vector v (df) with proper broadcasting
    if np.ndim(v) == 0:  # scalar
        # Calculate p(z=k|data)
        log_lik_matrix = (np.log(pi)[np.newaxis, :] +
                          v/2 * np.log(v/2) - gammaln(v/2) +
                          (v/2 - 1) * (2 * np.log(sehat))[:, np.newaxis] +
                          (alpha_vec * np.log(beta) - gammaln(alpha_vec) +
                           gammaln(alpha_vec + v/2))[np.newaxis, :] -
                          (alpha_vec + v/2)[np.newaxis, :] *
                          np.log(beta[np.newaxis, :] + (v/2 * sehat**2)[:, np.newaxis]))
        
        term2 = (alpha_vec + v/2) * modalpha_vec / (beta[np.newaxis, :] + (v/2 * sehat**2)[:, np.newaxis])
    else:  # vector
        v = np.asarray(v)
        v_half = v / 2
        # Calculate p(z=k|data)
        log_lik_matrix = (np.log(pi)[np.newaxis, :] +
                          (v_half * np.log(v_half) - gammaln(v_half) +
                           (v_half - 1) * (2 * np.log(sehat)))[:, np.newaxis] +
                          (alpha_vec * np.log(beta) - gammaln(alpha_vec) +
                           gammaln(alpha_vec[np.newaxis, :] + v_half[:, np.newaxis])) -
                          (alpha_vec[np.newaxis, :] + v_half[:, np.newaxis]) *
                          np.log(beta[np.newaxis, :] + (v_half[:, np.newaxis] * sehat[:, np.newaxis]**2)))
        
        term2 = ((alpha_vec[np.newaxis, :] + v_half[:, np.newaxis]) * modalpha_vec[np.newaxis, :] / 
                (beta[np.newaxis, :] + (v_half[:, np.newaxis] * sehat[:, np.newaxis]**2)))
    
    max_log = np.max(log_lik_matrix, axis=1, keepdims=True)
    lik_matrix = np.exp(log_lik_matrix - max_log)
    classprob = lik_matrix / np.sum(lik_matrix, axis=1, keepdims=True)
    
    term1 = alpha_vec / c
    
    gradmat = c * classprob * (term1[np.newaxis, :] - term2)
    grad = -np.sum(gradmat) # Negative because we minimize -loglike
    return grad

def loglike_se_ac(params, n, k, v, sehat, pi, unimodal):
    """Negative log-likelihood w.r.t. log(c) and log(alpha)."""
    c = np.exp(params[0])
    alpha_vec = np.exp(params[1:])
    if unimodal == 'variance':
        modalpha_vec = alpha_vec + 1
    else: # precision
        modalpha_vec = alpha_vec - 1
        if np.any(modalpha_vec <= 0): return np.inf # Constraint alpha > 1
        
    beta = c * modalpha_vec
    logl = loglike_vash(sehat, v, pi, alpha_vec, beta)
    
    val = -logl
    val = min(val, 1e200)
    val = max(val, -1e200)
    return val

def gradloglike_se_ac(params, n, k, v, sehat, pi, unimodal):
    """Gradient of negative log-likelihood w.r.t. log(c) and log(alpha)."""
    c = np.exp(params[0])
    alpha_vec = np.exp(params[1:])
    
    if unimodal == 'variance':
        modalpha_vec = alpha_vec + 1
        d_modalpha_d_alpha = 1.0
    else: # precision
        modalpha_vec = alpha_vec - 1
        d_modalpha_d_alpha = 1.0
        if np.any(modalpha_vec <= 0): return np.array([np.inf, np.inf])

    beta = c * modalpha_vec
    
    # Handle vector v (df) with proper broadcasting
    if np.ndim(v) == 0:  # scalar
        # Calculate p(z=k|data)
        log_lik_matrix = (np.log(pi)[np.newaxis, :] +
                          v/2 * np.log(v/2) - gammaln(v/2) +
                          (v/2 - 1) * (2 * np.log(sehat))[:, np.newaxis] +
                          (alpha_vec * np.log(beta) - gammaln(alpha_vec) +
                           gammaln(alpha_vec + v/2))[np.newaxis, :] -
                          (alpha_vec + v/2)[np.newaxis, :] *
                          np.log(beta[np.newaxis, :] + (v/2 * sehat**2)[:, np.newaxis]))
        
        denom = beta[np.newaxis, :] + (v/2 * sehat**2)[:, np.newaxis]
        term2_c = (alpha_vec + v/2) * modalpha_vec / denom
        term5 = -(alpha_vec + v/2) * (c * d_modalpha_d_alpha) / denom
    else:  # vector
        v = np.asarray(v)
        v_half = v / 2
        # Calculate p(z=k|data)
        log_lik_matrix = (np.log(pi)[np.newaxis, :] +
                          (v_half * np.log(v_half) - gammaln(v_half) +
                           (v_half - 1) * (2 * np.log(sehat)))[:, np.newaxis] +
                          (alpha_vec * np.log(beta) - gammaln(alpha_vec) +
                           gammaln(alpha_vec[np.newaxis, :] + v_half[:, np.newaxis])) -
                          (alpha_vec[np.newaxis, :] + v_half[:, np.newaxis]) *
                          np.log(beta[np.newaxis, :] + (v_half[:, np.newaxis] * sehat[:, np.newaxis]**2)))
        
        denom = beta[np.newaxis, :] + (v_half[:, np.newaxis] * sehat[:, np.newaxis]**2)
        term2_c = ((alpha_vec[np.newaxis, :] + v_half[:, np.newaxis]) * modalpha_vec[np.newaxis, :] / denom)
        term5 = -((alpha_vec[np.newaxis, :] + v_half[:, np.newaxis]) * (c * d_modalpha_d_alpha) / denom)
    
    max_log = np.max(log_lik_matrix, axis=1, keepdims=True)
    lik_matrix = np.exp(log_lik_matrix - max_log)
    classprob = lik_matrix / np.sum(lik_matrix, axis=1, keepdims=True)

    # Gradient w.r.t log(c)
    term1_c = alpha_vec / c
    gradmat_c = c * classprob * (term1_c[np.newaxis, :] - term2_c)
    grad_c = -np.sum(gradmat_c)

    # Gradient w.r.t log(alpha)
    d_logbeta_d_logalpha = alpha_vec * d_modalpha_d_alpha / modalpha_vec
    
    grad_a_term1 = np.log(beta) + d_logbeta_d_logalpha
    grad_a_term2 = -digamma(alpha_vec)
    if np.ndim(v) == 0:  # scalar
        grad_a_term3 = digamma(alpha_vec + v/2)
    else:  # vector
        grad_a_term3 = digamma(alpha_vec[np.newaxis, :] + v_half[:, np.newaxis])
    grad_a_term4 = -np.log(denom)
    
    grad_a_mat = classprob * (grad_a_term1 + grad_a_term2 + grad_a_term3 + grad_a_term4 + term5)
    grad_a = -np.sum(alpha_vec * np.sum(grad_a_mat, axis=0))

    res = np.array([grad_c, grad_a])
    res = np.minimum(res, 1e200)
    res = np.maximum(res, -1e200)
    return res

# ==============================================================================
# Helper and Posterior Calculation Functions
# ==============================================================================

def est_singlecomp_mm(sehat, df):
    """Estimates single component inverse-gamma prior by method of moments."""
    n = len(sehat)
    
    # Handle vector df by using its mean
    df_scalar = np.mean(df) if np.ndim(df) > 0 else df
    
    e = 2 * np.log(sehat) - digamma(df_scalar / 2) + np.log(df_scalar / 2)
    ehat = np.mean(e)
    # n/(n-1) is Bessel's correction for variance
    var_e = np.var(e, ddof=1) if n > 1 else 0
    
    trigamma_a = var_e - polygamma(1, df_scalar / 2)
    a = solve_trigamma(trigamma_a)
    
    b = a * np.exp(ehat + digamma(df_scalar / 2) - np.log(df_scalar / 2))
    return {'a': a, 'b': b}

def solve_trigamma(x):
    """Solves polygamma(1, y) = x for y."""
    if x > 1e7:
        return 1 / np.sqrt(x)
    if x < 1e-6:
        return 1 / x
    if x <= 0: # Trigamma is always positive, so no solution for y > 0
        return np.nan

    # Newton-Raphson method
    y = 0.5 + 1 / x
    for _ in range(10): # Iterate a few times
        y_old = y
        f = polygamma(1, y) - x
        f_prime = polygamma(2, y)
        delta = -f / f_prime
        y += delta
        if np.abs(delta) < 1e-8 * np.abs(y_old):
            break
    return y

def post_igmix(m: Igmix, sebetahat: np.ndarray, v: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
    """Computes posterior shape and rate parameters."""
    n = len(sebetahat)
    
    # Handle vector v (df) with proper broadcasting
    if np.ndim(v) == 0:  # scalar
        alpha1 = np.tile(m.alpha + v/2, (n, 1))
        beta1 = m.beta[np.newaxis, :] + (v/2 * sebetahat**2)[:, np.newaxis]
    else:  # vector
        v = np.asarray(v)
        alpha1 = m.alpha[np.newaxis, :] + (v/2)[:, np.newaxis]
        beta1 = m.beta[np.newaxis, :] + ((v/2)[:, np.newaxis] * sebetahat[:, np.newaxis]**2)
    
    return {'alpha': alpha1, 'beta': beta1}

def comp_postprob(g: Igmix, data: Dict[str, np.ndarray]) -> np.ndarray:
    """Computes posterior probability of component membership, P(Z=k|data)."""
    pi_mat = post_pi_vash(
        A=get_A(len(data['s']), g.ncomp(), data['v'], g.alpha, g.beta / g.c, data['s']),
        n=len(data['s']), k=g.ncomp(), v=data['v'], sehat=data['s'],
        alpha_vec=g.alpha, modalpha_vec=g.beta / g.c, c=g.c, pi=g.pi
    )
    return pi_mat / np.sum(pi_mat, axis=1, keepdims=True)

def mod_t_test(betahat: np.ndarray, se: np.ndarray, pi: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Moderated t-test with standard errors moderated by mixture prior."""
    n, k = pi.shape
    pvalue = np.full(n, np.nan)
    
    complete_obs = ~np.isnan(betahat) & ~np.any(np.isnan(se), axis=1)
    n_complete = np.sum(complete_obs)
    
    if n_complete > 0:
        t_stat = betahat[complete_obs, np.newaxis] / se[complete_obs, :]
        temp_pvalue = t_dist.cdf(t_stat, df=df[complete_obs, :])
        temp_pvalue = np.minimum(temp_pvalue, 1 - temp_pvalue) * 2
        pvalue[complete_obs] = np.sum(pi[complete_obs, :] * temp_pvalue, axis=1)
        
    return pvalue

def mod_t_test_one_side(betahat: np.ndarray, se: np.ndarray, pi: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Moderated t-test with standard errors moderated by mixture prior."""
    n, k = pi.shape
    pvalue = np.full(n, np.nan)
    
    complete_obs = ~np.isnan(betahat) & ~np.any(np.isnan(se), axis=1)
    n_complete = np.sum(complete_obs)
    
    if n_complete > 0:
        t_stat = betahat[complete_obs, np.newaxis] / se[complete_obs, :]
        temp_pvalue = t_dist.cdf(t_stat, df=df[complete_obs, :])
        temp_pvalue = temp_pvalue * 2
        pvalue[complete_obs] = np.sum(pi[complete_obs, :] * temp_pvalue, axis=1)
        
    return pvalue


# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == '__main__':
    print("## Running simple example from R documentation ##")
    np.random.seed(42)
    # Generate true variances (sd^2) from an inverse-gamma prior
    # R's rgamma uses shape and rate; numpy uses shape and scale (scale=1/rate)
    true_precision = np.random.gamma(shape=5, scale=1/5, size=100)
    true_sd = np.sqrt(1 / true_precision)
    # Observed standard errors are estimates of true sd's
    sehat_obs = np.sqrt(true_sd**2 * np.random.chisquare(df=7, size=100) / 7)
    
    # Run the vash function with vector df (all values = 7)
    df_vector = np.full(100, 7.0)
    fit = vash(sehat_obs, df=df_vector)
    
    print("\nFitted Prior (g):")
    print(fit['fitted_g'])
    
    print("\nFirst 5 observed vs. posterior standard deviations:")
    print("Observed:", np.round(sehat_obs[:5], 4))
    print("Posterior:", np.round(fit['sd_post'][:5], 4))
    
    plt.figure(figsize=(6, 6))
    plt.scatter(sehat_obs, fit['sd_post'], alpha=0.6)
    plt.plot([0, max(sehat_obs)], [0, max(sehat_obs)], 'r--', label='y=x')
    plt.xlabel("Observed Standard Errors (sehat)")
    plt.ylabel("Posterior Estimated Standard Deviations")
    plt.title("Vash Shrinkage Results (Vector DF)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.legend()
    print("\nPlotting results...")
    plt.show()
        
    print("\n" + "="*50 + "\n")
    
    print("## Running example with a pre-specified g and vector df ##")
    # Generate data from two components
    true_precision1 = np.random.gamma(shape=5, scale=1/4, size=50)
    true_precision2 = np.random.gamma(shape=10, scale=1/9, size=50)
    true_sd_g = np.sqrt(1 / np.concatenate([true_precision1, true_precision2]))
    sehat_obs_g = np.sqrt(true_sd_g**2 * np.random.chisquare(df=7, size=100) / 7)
    
    # Define the "true" g
    true_g = Igmix(pi=np.array([0.5, 0.5]), 
                   alpha=np.array([5, 10]), 
                   beta=np.array([4, 9]))
                   
    print("True g:", true_g)
    
    # Run vash with the specified g and vector df
    df_vector_g = np.full(100, 7.0)
    fit_g = vash(sehat_obs_g, df=df_vector_g, g=true_g)
    
    print("\nFitted g (should be same as true_g as EM runs for 1 iter):")
    print(fit_g['fitted_g'])
    
    print("\nFirst 5 observed vs. posterior standard deviations (with true g and vector df):")
    print("Observed:", np.round(sehat_obs_g[:5], 4))
    print("Posterior:", np.round(fit_g['sd_post'][:5], 4))
    
    print("\n" + "="*50 + "\n")
    
    print("## Running example with varying degrees of freedom ##")
    # Create a vector with varying df values
    df_varying = np.random.choice([5, 7, 9], size=100, p=[0.3, 0.4, 0.3])
    
    # Generate data with varying df
    sehat_varying = []
    for i, df_val in enumerate(df_varying):
        sehat_varying.append(np.sqrt(true_sd[i]**2 * np.random.chisquare(df=df_val, size=1)[0] / df_val))
    sehat_varying = np.array(sehat_varying)
    
    # Run vash with varying df
    fit_varying = vash(sehat_varying, df=df_varying)
    
    print(f"\nDF values range: {np.min(df_varying)} to {np.max(df_varying)}")
    print("Fitted Prior (g) with varying df:")
    print(fit_varying['fitted_g'])
    
    print("\nFirst 5 observed vs. posterior standard deviations (varying df):")
    print("Observed:", np.round(sehat_varying[:5], 4))
    print("Posterior:", np.round(fit_varying['sd_post'][:5], 4))
    print("DF values:", df_varying[:5])