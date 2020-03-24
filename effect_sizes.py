# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:14:52 2020

@author: L

TODO: AnovaRM needs to output SSeffect and SSerror for each factor
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
from scipy.stats import norm

#~Testing Dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Artificial movie ratings dataset
movie_1 = [9, 7, 8, 9, 8, 9, 9, 10, 9, 9]
movie_2 = [9, 6, 7, 8, 7, 9, 8, 8, 8, 7]
df = pd.DataFrame({'movie_1': movie_1,
                   'movie_2': movie_2})
df['difference'] = df['movie_1'] - df['movie_2']
print(df)

# Useful for testing functions
a, b = movie_1, movie_2
na, nb = np.size(a), np.size(b)

#~Scenario 1: Independent Groups ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
t, p, dof = sm.stats.ttest_ind(movie_1, movie_2)
print(f"t({dof}) = {t:.2f}")

# Standardised Cohen's d with the pooled s


def standardised_cohens_d(a, b):
    na, nb = np.size(a), np.size(b)  # Get the n
    mean_a, mean_b = np.mean(a), np.mean(b)  # get the means
    Sa, Sb = np.std(a, ddof=1), np.std(b, ddof=1)  # Get the sample S
    numerator = mean_a - mean_b
    denominator = np.sqrt(((na - 1) * Sa**2 + (nb - 1) * Sb**2) / (na + nb - 2))

    return numerator / denominator


# Sample-based estimate: not recommended for meta-analytic work because biased by sample averages
d_s = standardised_cohens_d(movie_1, movie_2)
print(f"Standardised cohen's d: {d_s:.2f}")


# Useful in later meta-analytic work
def standardised_hedges_g(a, b):
    # ***Hedge's g(s) depends on cohen's d(s) but overcomes the bias
    # Small difference but apparently preferable.
    d_s = standardised_cohens_d(a, b)
    na, nb = np.size(a), np.size(b)
    return d_s * (1 - (3 / (4 * (na + nb) - 9)))


hg_s = standardised_hedges_g(a, b)
print(f"Hedge's g: {hg_s:.2f}")


# Extra: "Common language" effect size (CLES)
# Probability that person from a will have a higher est. than person from b (or equv for within-subjs)
def common_language_effect(a, b=None):
    if b == None:
        # We assume related samples and that a represents the difference values
        # McGraw & Wong (1992)
        Z = np.mean(a) / np.std(a, ddof=1)
    else:
        d_s = standardised_cohens_d(a, b)
        Z = d_s / np.sqrt(2)
    CLES = norm.cdf(Z)
    return CLES


cles = common_language_effect(a, b)
print(f"Common language effect size: {cles:.2f}")


#~Scenario 2: Within-subjects design ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
from statsmodels.stats import weightstats as sms

corrcoef = np.corrcoef(a, b)
r = corrcoef[0][1]
print(f"Strong correlation within-subs: {r:.2f}")

# Strong correlation means small standard dev for difference scores
d = sms.DescrStatsW(np.array(a) - np.array(b))
t2, p2, dof2 = d.ttest_mean()
print(f"t({dof2:.2f}) = {t2:.2f}, p = {p2:.2f}")


def cohens_d_of_diff(a, b, alt=True):
    if alt == False:
        # Rarely used in meta analysis, because in those cases we compare both within and between designs
        # This only allows within.
        d_z = np.mean(np.array(a) - np.array(b)) / np.std(np.array(a) - np.array(b), ddof=2)
    else:
        # Correction to enable comparison of d across within and between designs
        d_z = (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) - (2 * r * np.std(a, ddof=1) * np.std(b, ddof=1)))  # * np.sqrt(2 * (1 - r))
    return d_z


diff = np.array(a) - np.array(b)
d_z1 = cohens_d_of_diff(a, b, alt=False)
d_z2 = cohens_d_of_diff(a, b, alt=True)
print(f"Cohens d on the difference: {d_z1:.2f} (corrected: {d_z2:.2f})")

#~Scenario 3: ANOVA approach ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#
dfs = df[['movie_1', 'movie_2']].stack().reset_index()  # stacked
dfs.columns = ['subject', 'movie', 'rating']
lm = ols('rating ~ movie', data=dfs).fit()
result = sm.stats.anova_lm(lm)

F = result.loc['movie']['F']
dof_numer = result.loc['movie']['df']
dof_denom = result.loc['Residual']['df']
p = result.loc['movie']['PR(>F)']
print(f"F({dof_numer:.2f}, {dof_denom:.2f}) = {F:.2f}, p = {p:.2f}")

# We need the sums of squares
SSeffect, SSerror = result['sum_sq']


def eta_squared(SSeffect, SSerror):
    return SSeffect / (SSeffect + SSerror)


etasq = eta_squared(SSeffect, SSerror)
print(f"η^2: {etasq:.2f}")

# Convert cohen's d to R
d = d_s
N = dfs.shape[0]
n1 = dfs.loc[dfs['movie'] == 'movie_1'].shape[0]
n2 = dfs.loc[dfs['movie'] == 'movie_2'].shape[0]
rpb = d / np.sqrt(d**2 + ((N**2 - N * 2) / (n1 * n2)))
# this is the r value, so squaring gives r^2: variance explained
print(f"rpb: {rpb:.2f}, r^2: {rpb**2:.2f} <--- same as eta squared.")

###
anova = AnovaRM(data=dfs, subject='subject', within=['movie'], depvar='rating')
fit = anova.fit()
res = fit.anova_table
F = res['F Value'].squeeze()
dof_num = res['Num DF'].squeeze()
dof_den = res['Den DF'].squeeze()
p = res['Pr > F'].squeeze()
print(f"F({dof_num:.2f}, {dof_den:.2f}) = {F:.2f}, p = {p:.2f}")

print(fit.summary())
