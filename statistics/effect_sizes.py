# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:47:12 2020

@author: L

there are at least 5 different and non-equivalent ways that people might 
compute a d-like effect size (which they would invariably simply call “Cohen’s d”) 
... and the resulting effect sizes range from about 0.25 to 1.91.

Useful links
https://cran.r-project.org/web/packages/effectsize/effectsize.pdf
http://jakewestfall.org/blog/index.php/2016/03/25/five-different-cohens-d-statistics-for-within-subject-designs/
http://jeffrouder.blogspot.com/2016/03/the-effect-size-puzzler-answer.html
https://forum.cogsci.nl/discussion/3013/what-denominator-does-the-cohens-d-use-on-jasp
https://www.uv.es/uvetica/files/Cunningham_McCrum_Gardner2007.pdf
https://github.com/mtorchiano/effsize/blob/master/R/CohenD.R
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats import weightstats as sms
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
from scipy import stats




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


#~ Dependent Groups / Within Samples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def cohens_d(x1=None, x2=None, S_x1=None, S_x2=None, ind=True, r=None, hedges_correction=False, nobs_x1=None, nobs_x2=None):
    
    realval_check = (int, float, np.float16, np.float32, np.float64)
    if isinstance(x1, realval_check) and isinstance(x2, realval_check):
        x1, x2 = [x1], [x2]
    
        
    x1, x2 = np.array(x1), np.array(x2)
    mean_x1, mean_x2 = np.mean(x1), np.mean(x2) # if means provided as x1,x2, then these will return the same vals.
    meandiff = mean_x1 - mean_x2
    
    if len(x1)==1 and len(x2)==1:
        # Then just the means were provided: we need both SDs otherwise error.
        if not (S_x1 and S_x2):
            raise ValueError("Requires both of `S_x1` & `S_x2`.")
        else:
            from_summary = True
    else:
        # Then x1 and x2 are arrays and we proceed as usual
        from_summary = False
        S_x1, S_x2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        nobs_x1, nobs_x2 = len(x1), len(x2)
    
    if ind == True:
        # Independent samples        
        if (from_summary == True) and (not (nobs_x1 and nobs_x2)):
            # Estimating using different pooled SD e.g. https://www.polyu.edu.hk/mm/effectsizefaqs/effect_size_equations2.html
            pooled_sd = np.sqrt((S_x1**2 + S_x2**2) / 2)
            print(pooled_sd)
        else:
            
            dof = nobs_x1 + nobs_x2 - 2
            pooled_sd = np.sqrt( ((nobs_x1 - 1) * S_x1 ** 2 + (nobs_x2 - 1) * S_x2 ** 2) / dof )
            
    else:
        # Related samples
        if from_summary == True:
            if not r: 
                # We need the coefficient otherwise it is biased. Maybe just raise warning and use biased cohens d?
                raise ValueError("Correlation coefficient `r` required to compute effect size from summary statistics, but was not provided.")
        else:
            # Need to compute r from x1 and x2 arrays
            r = np.corrcoef(x1, x2)[0][1]
            
        pooled_sd = np.sqrt(S_x1**2 + S_x2**2 - 2 * r * S_x1 * S_x2)
        # pooled_sd = np.mean([S_x1, S_x2]) # Hedges' g (average)
     
    # Calculate the effect size
    effectsize = meandiff / pooled_sd
    
    if hedges_correction:
        if (from_summary == True) and (not (nobs_x1 and nobs_x2)):
            raise ValueError("Requires number of observations from both groups in order to apply Hedges' correction, but these were not provided.")
        else:
            if ind == False:
                # Re-calculate the effect size with the average pooled stdev
                effectsize = meandiff / np.mean([S_x1, S_x2])
            
            effectsize *= (1 - (3 / (4 * (nobs_x1 + nobs_x2) - 9)))
    
    return effectsize
    


# Cohen's d calcs
d_s_ind = cohens_d(x1, x2)
d_s_rel = cohens_d(x1, x2, ind=False)
d_s_ind_summary = cohens_d(np.mean(x1), np.mean(x2), np.std(x1, ddof=1), np.std(x2, ddof=1))
d_s_rel_summary = cohens_d(np.mean(x1), np.mean(x2), np.std(x1, ddof=1), np.std(x2, ddof=1), r=0.73)

# Hedges' g correction for Cohen's d
d_s_ind_g = cohens_d(x1, x2, hedges_correction=True)
d_s_rel_g = cohens_d(x1, x2, ind=False, hedges_correction=True)
d_s_ind_summary_g = cohens_d(np.mean(x1), np.mean(x2), np.std(x1, ddof=1), np.std(x2, ddof=1), 
                             nobs_x1=10, nobs_x2=10, hedges_correction=True)
d_s_rel_summary_g = cohens_d(np.mean(x1), np.mean(x2), np.std(x1, ddof=1), np.std(x2, ddof=1), 
                             nobs_x1=10, nobs_x2=10, r=0.73, hedges_correction=True)

# Test them against the examples
np.testing.assert_allclose(round(d_s_ind, 2), 1.13) # from the paper
np.testing.assert_allclose(round(d_s_rel, 2), 1.5) # from the paper
np.testing.assert_allclose(round(d_s_ind_summary, 2), 1.13) # tested with g*power
np.testing.assert_allclose(round(d_s_rel_summary, 2), 1.13) # tested from here ---> https://www.socscistatistics.com/effectsize/default3.aspx
np.testing.assert_allclose(round(d_s_ind_g, 2), 1.08) # from the paper
np.testing.assert_allclose(round(d_s_rel_g, 2), 1.08) # from the paper
np.testing.assert_allclose(round(d_s_ind_summary_g, 2), 1.08) # tested from here ---> https://www.socscistatistics.com/effectsize/default3.aspx
np.testing.assert_allclose(round(d_s_rel_summary_g, 2), 1.08) # tested from here ---> https://www.socscistatistics.com/effectsize/default3.aspx


print(f"Cohens d from independent samples data: {d_s_ind}")
print(f"Cohens d from paired samples data: {d_s_rel}")
print(f"Cohens d from independent samples data using summary stats: {d_s_ind_summary}")
print(f"Cohens d from paired samples data using summary stats: {d_s_rel_summary}")

print(f"Hedges g from independent samples data: {d_s_ind_g}")
print(f"Hedges g from paired samples data: {d_s_rel_g}")
print(f"Hedges g from independent samples data using summary stats: {d_s_ind_summary_g}")
print(f"Hedges g from paired samples data using summary stats: {d_s_rel_summary_g}")

print(df.describe())
