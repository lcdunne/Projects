# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:39:58 2020

@author: L
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
from scipy.stats import norm

'''
Hedge's g involves much of the same computation as cohen's d, and even uses
Cohen's d in its calculation...
    Separate the function into two with repetitive code?
    Calculate both in a more vague function?
'''

# def _effect_size_ind() <-- to be used in ttest_ind()?

def standardised_cohens_d_hedges_g(x1=None, x2=None, t=None, nobs_x1=None, nobs_x2=None, N=None):
    '''
    Computes the standardised Cohen's d effect size statistic for independent samples
    using the pooled standard deviation.
    
    Cliff delta would be the alternative for nonparametric

    Parameters
    ----------
    x1 : array-like, optional
        first of the two independent samples. The default is None.
    x2 : array-like, optional
        second of the two independent samples. The default is None.
    tstat : scalar, optional
        t-statistic result from independent t-test. The default is None.
    nobs_x1 : scalar, optional
        number of observations for the first of the two independent samples. The default is None.
    nobs_x2 : scalar, optional
        number of observations for the second of the two independent samples. The default is None.
    N : scalar, optional
        total N (sum of both independent samples). 
        Note that if only `tstat` and `N` are provided, the standardised 
        Cohen's d is an approximate. The default is None.

    Raises
    ------
    ValueError
        When x1 and x2 are None, if any of the following are true:
            t == None
            (nobs_x1 == None) and (nobs_x2 == None) and (N == None)

    Returns
    -------
    cohens_d : float
        Standardised Cohen's d statistic.

    '''
    # if x1 and x2 are provided - compute d from this.
    if (x1 and x2):
        x1, x2 = np.array(x1), np.array(x2) # Ensure np.array
        nobs_x1, nobs_x2 = len(x1), len(x2)
        S_x1, S_x2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        dof = nobs_x1 + nobs_x2 - 2 # compute degree of freedom
        meandiff = np.mean(x1) - np.mean(x2)
        pooled_sd = np.sqrt( ((nobs_x1 - 1) * S_x1 ** 2 + (nobs_x2 - 1) * S_x2 ** 2) / dof )
        cohens_d = meandiff / pooled_sd
        hedges_g = cohens_d * (1 - (3 / (4 * (nobs_x1 + nobs_x2) - 9)))
        
    elif (not x1 and not x2):
        # neither were given, so can we calculate from t and n obs?
        if t==None:
            raise ValueError("If `x1` and `x2` are not provided, `t` is needed to compute Cohen's d but was not provided.")
        else:
            # We have the t value
            # Do we have both observation counts?
            if not nobs_x1 and not nobs_x2:
                if not N:
                    raise ValueError("Either `N` or both of `nobs_x1` & `nobs_x2` is required but was not provided.")
                else:
                    # Estimating
                    cohens_d = 2 * t / np.sqrt(N)
                    hedges_g = cohens_d * (1 - (3 / (4 * N - 9)))
            else:
                # We have both n obs
                cohens_d = t * np.sqrt( (1 / nobs_x1) + (1 / nobs_x2) )
                hedges_g = cohens_d * (1 - (3 / (4 * (nobs_x1 + nobs_x2) - 9)))
        
    return cohens_d, hedges_g

def common_language_effect_size(x1=None, x2=None):
    '''
    Common language effect size McGraw & Wong (1992) reflects the probability 
    that a randomly sampled observation from one sample will have a higher 
    observed measurement than a randomly sampled observation from the other 
    sample, either between or within.

    Parameters
    ----------
    x1 : array-like, optional
        first of the two independent samples. If `x2` not specified, `x1` is 
        inferred to be a vector representing the difference between the two 
        samples. The default is None.
    x2 : array-like, optional
        second of the two independent samples. The default is None.

    Raises
    ------
    ValueError
        When no inputs are given.

    Returns
    -------
    CLES : float
        The common language effect size.

    '''
    if (x1==None and x2 == None):
        raise ValueError("Requires `x1` and `x2`, or difference between them as `x1`, but nothing was provided.")
    elif x2 == None:
        # We assume related samples and that x1 represents the difference values
        Z = np.mean(x1) / np.std(x1, ddof=1)
    else:
        cohens_d, _ = standardised_cohens_d_hedges_g(x1, x2)
        Z = cohens_d / np.sqrt(2)
        
    CLES = norm.cdf(Z)
    return CLES

#~Testing Dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Artificial movie ratings dataset
x1 = [9, 7, 8, 9, 8, 9, 9, 10, 9, 9]
x2 = [9, 6, 7, 8, 7, 9, 8, 8, 8, 7]
df = pd.DataFrame({'movie_1': x1,
                   'movie_2': x2})
df['difference'] = df['movie_1'] - df['movie_2']
dfs = df[['movie_1', 'movie_2']].stack().reset_index()
print(df, '\n')

nobs_x1, nobs_x2 = np.size(x1), np.size(x2)


#~ Independent Groups ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
t, p, dof = sm.stats.ttest_ind(x1, x2)
print(f"t({dof}) = {t:.2f}, p = {p:.2f}\n")

d1, g1 = standardised_cohens_d_hedges_g(x1=x1, x2=x2)
d2, g2 = standardised_cohens_d_hedges_g(t=t, nobs_x1=nobs_x1, nobs_x2=nobs_x2)
d3, g3 = standardised_cohens_d_hedges_g(t=t, N=nobs_x1 + nobs_x2)

cles = common_language_effect_size(x1=x1,x2=x2)

# Test them against the example from the paper
np.testing.assert_allclose(round(d1, 2), 1.13)
np.testing.assert_allclose(round(g1, 2), 1.08)

np.testing.assert_allclose(round(d2, 2), 1.13)
np.testing.assert_allclose(round(g2, 2), 1.08)

np.testing.assert_allclose(round(d3, 2), 1.13)
np.testing.assert_allclose(round(g3, 2), 1.08)

print("Testing standardised_cohens_d() for different inputs:")
print(f"""Using x1 & x2:\n\td = {d1}\n\tg = {g1}\nUsing t, nobs_x1 & nobs_x2:\n\td = {d2}\n\tg = {g2}\nUsing just t and N\n\td = {d3}\n\tg = {g3}""")
np.testing.assert_allclose(round(cles, 2), 0.79)
print(f"Testing the CLES:\n\tCLES = {cles}")

