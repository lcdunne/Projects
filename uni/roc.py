import numpy as np
# import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
np.seterr(all='ignore')


def accumulate(*arrays):
    accumulated = []
    for a in arrays:
        accum = np.cumsum(a)
        accumulated.append(accum)

    # In case it's just a single array pop it out
    if len(accumulated) == 1:
        accumulated = accumulated.pop()

    return accumulated


def roc(*arrays, truncate=False):
    # Assumes inputs are observed counts at each decision point on scale
    rates = []
    for a in arrays:
        # accumulate it then calculate it
        accum = accumulate(a)
        frequency = np.array([(x + i / len(accum)) / (max(accum) + 1) for i, x in enumerate(accum, start=1)])
        if truncate:
            # trim off the last element
            frequency = frequency[:-1]
        rates.append(frequency)

    if len(rates) == 1:
        rates = rates.pop()

    return rates


def z_score(*arrays):
    # compute the z score of ROC data
    z_arrays = []
    for a in arrays:
        z_arrays.append(stats.norm.ppf(a))

    if len(z_arrays) == 1:
        z_arrays = z_arrays.pop()

    return z_arrays


def high_threshold(R=None, noise_roc=None):
    # R will become the intercept on the ROC plot
    # noise_roc is optional;
    #    if provided, then it should be roc-ified & truncated with roc(noise, truncate=True)
    if noise_roc is None:
        # theoretical
        noise_roc = np.array([0, 1])

    # Enforce numpy array
    noise_roc = np.array(noise_roc)

    # Compute threshold x and y
    x = noise_roc
    y = (1 - R) * noise_roc + R

    return x, y


def signal_detection(d=None, c=None, signal_var=1):
    if c is None:
        # theoretical
        c = np.linspace(-5, 5, 500)
    # Enforce numpy
    c = np.array(c)

    # Compute signal detection x & y with optional unequal variance
    x = stats.norm.cdf(-d / 2 - c)  # noise distribution
    y = stats.norm.cdf(d / 2 - c, scale=signal_var)  # signal distribution

    return x, y


def dual_process(R=None, d=None, c=None):
    if c is None:
        # theoretical
        c = np.linspace(-5, 5, 500)
    # Enforce numpy
    c = np.array(c)

    # Compute dual-process (signal detection + high threshold) x & y
    x = stats.norm.cdf(-d / 2 - c)
    y = R + (1 - R) * stats.norm.cdf(d / 2 - c)

    return x, y


def strength_estimate(d=None, c=None):
    # E.g. recognition familiarity
    c = np.array(c)
    cut_point = int(np.median(np.arange(0, len(c))))
    return stats.norm.cdf(d / 2 - c[cut_point])


def g_test(x, x_freq, expected, x_max):
    # Refs: [1, 2]
    # Two-way log-likelihood G-test
    # Implementation issues:
    #    depending on minimization starting variables, the log expressions can throw errors from negative values and div/0 etc. Numpy just warns and continues.
    a = 2 * x * np.log(x_freq / expected)
    b = 2 * (x_max - x) * np.log((1 - x_freq) / (1 - expected))
    return a + b


def sum_g_squared(parameters, labels, signal, noise, model='sdt', optimizing=False):
    # Formerly called `detection_model()`
    # parameters: variable parameters to adjust if optimizing. Must be list even if size of 1
    # labels: string names corresponding to each parameters. MUST be of equal size to parameters, also list
    # signal: observed counts at each decision point on scale for SIGNAL PRESENT trials
    # noise: observed counts at each decision point on scale for SIGNAL ABSENT trials
    # model: optional, can be any one of 'ht' (high threshold), 'sdt' (signal detection theory), 'dpsdt' (dual process + signal detection).
    # optimizing: no longer needed optional, default False. When set to true, only the sum of g squared is returned

    assert len(parameters) == len(labels), print(f"Length of parameters ({len(parameters)}) does not match length of labels ({len(labels)}).")
    # Ensure parameter and labels are numpy
    parameters = np.array(parameters)
    labels = np.array(labels)

    # Get the ROC data and truncate it ready for computing sum of g squared (avoids values of 0 or 1)
    signal_roc, noise_roc = roc(signal, noise, truncate=True)
    # Get the accumulated inputs as well but store the maximum value before truncating.
    signal_acc, noise_acc = accumulate(signal, noise)
    # The maximum value should always be last after the accumulate function is applied, so we just index it.
    signal_acc_max = signal_acc[-1]  # extract the maximum
    signal_acc = signal_acc[:-1]  # truncate signal_acc
    noise_acc_max = noise_acc[-1]
    noise_acc = noise_acc[:-1]

    if model == 'ht':
        # High Threshold. labels `variable` MUST contain `R`, with corresponding `parameters` value of R
        if ('R' not in labels):
            raise ValueError(f"`R`not specified in parameter labels. labels specified were: {labels}.")
        R = parameters[labels == 'R'].item()
        noise_expected, signal_expected = high_threshold(R=R, noise_roc=noise_roc)

    elif model in ['sdt', 'uvsdt']:
        if 'd' not in labels:
            raise ValueError(f"Either or both of `R` and `d` not specified in parameter labels. labels specified were: {labels}.")
        # Check if signal variance specified, otherwise just keep it at default. determine c parameters
        if 'signal_var' in labels:
            signal_var = parameters[labels == 'signal_var']
            c = parameters[(labels != 'R') & (labels != 'd') & (labels != 'signal_var')]
        else:
            # Explicitly keeping at default var
            signal_var = 1
            c = parameters[(labels != 'R') & (labels != 'd')]
        d = parameters[labels == 'd'].item()
        noise_expected, signal_expected = signal_detection(d=d, c=c, signal_var=signal_var)

    elif model == 'dpsdt':
        if ('R' not in labels) or ('d' not in labels):
            raise ValueError(f"`R` not specified in parameter labels. labels specified were: {labels}.")
        # Grab the R and d values
        R = parameters[labels == 'R'].item()
        d = parameters[labels == 'd'].item()

        c = parameters[(labels != 'R') & (labels != 'd')]
        # Get the expected values for signals and noises
        noise_expected, signal_expected = dual_process(R, d, c)

    # compute g squared values for signal and noise data
    signal_g_squared = g_test(x=signal_acc, x_freq=signal_roc, expected=signal_expected, x_max=signal_acc_max)
    noise_g_squared = g_test(x=noise_acc, x_freq=noise_roc, expected=noise_expected, x_max=noise_acc_max)

    sum_of_g_squared = np.sum([signal_g_squared, noise_g_squared])

    if optimizing:
        return sum_of_g_squared
    else:
        return sum_of_g_squared, noise_expected, signal_expected

def optimize_model(objective, labels, signal, noise, model, iterations=100):

    for i in range(1, int(iterations + 1)):
        if i == 1:
            parameters = np.zeros(len(labels))  # if other testing is shit then might need to rethink this
            parameters[labels == 'R'] = 0.99  # Make sure R has a good starting value (if it exists)
        else:
            parameters[labels != 'R'] = np.random.random_sample(len(parameters[labels != 'R']))

        # print(f"Starting parameters:\n\t\t{parameters}")
        optimizing = True  # requires True to suppress multiple return values from the objective function

        opt = minimize(fun=objective, x0=parameters, args=(labels, signal, noise, model, optimizing), tol=1e-4)

        if opt.success:
            break

    opt_parameters = opt.x
    status = "successful" if opt.success else "failed"
    print(f"{model} optimization {status} over {i} iterations")
    print(f"{status} starting parameters:{parameters}")
    print(f"{status} final parameters:{opt_parameters}")
    print(f"{status} G\N{SUPERSCRIPT TWO}: {opt.fun}\n")

    # Get the labels & values corresponding to NON decision variables i.e. NOT c values
    non_decision_labels = labels[np.isin(labels, ['R', 'd', 'signal_var'])]
    non_decision_parameters = opt_parameters[np.isin(labels, ['R', 'd', 'signal_var'])]

    # Get the decision labels and values
    decision_labels = labels[~np.isin(labels, ['R', 'd', 'signal_var'])]
    decision_parameters = opt_parameters[~np.isin(labels, ['R', 'd', 'signal_var'])]

    output = {'model': model, 'optimize_result': opt,
              'data': {'signal': signal, 'noise': noise}, 'minimum': opt.fun, 'labels': labels,
              'c_vars': decision_parameters, 'c_labels': decision_labels,
              'non_c_vars': {l: p for l, p in zip(non_decision_labels, non_decision_parameters)}}

    return output


def fit_model(signal, noise, model):
    if type(model) == str:
        # Enforce model name to be list
        model = [model]

    # Ensure model names is correctly specified
    for m in model:
        if m == 'ht':
            parameter_names = np.array(['R'])
        elif m == 'sdt':
            parameter_names = np.array(['d', 1, 2, 3, 4, 5])
        elif m == 'uvsdt':
            parameter_names = np.array(['signal_var', 'd', 1, 2, 3, 4, 5])
        elif m == 'dpsdt':
            parameter_names = np.array(['R', 'd', 1, 2, 3, 4, 5])

        opt_models = {}

        opt_model = optimize_model(objective=sum_g_squared,
                                   labels=parameter_names,
                                   signal=signal,
                                   noise=noise,
                                   model=m)
        opt_models[m] = opt_model

    return opt_models


signal = [431, 218, 211, 167, 119, 69]
noise = [102, 161, 288, 472, 492, 308]
dpsdt_fit = fit_model(signal=signal, noise=noise, model='dpsdt')
dpsdt_fit


if __name__ == '__main__':

    signal, noise = [13, 10, 12, 11, 9, 1], [4, 4, 22, 35, 15, 4]
    print(f"Original data:\n\tSignal ratings: {signal}\n\tNoise ratings:{noise}\n")
    print(f"Accumulated data:\n\t{accumulate(signal, noise)}\n")
    print(f"ROC data:\n\t{roc(signal, noise)}\n")
    print(f"Z-score of ROC data:\n\t{z_score(roc(signal, noise))}\n")

    signal = [431, 218, 211, 167, 119, 69]
    noise = [102, 161, 288, 472, 492, 308]

    for model_name in ['ht', 'sdt', 'uvsdt', 'dpsdt']:

        model_fit = fit_model(signal=signal, noise=noise, model=model_name)

       # Get the observed ROC data
        roc_signal, roc_noise = roc(signal, noise, truncate=True)

        # Get the modelled data
        # Get the optimized likelihood
        sumg2 = model_fit[model_name]['minimum']

        if model_name != 'ht':
            # Get the c values for estimating familiarity
            c_vars = model_fit[model_name]['c_vars']

        if model_name in ['ht', 'dpsdt']:
            R = model_fit[model_name]['non_c_vars']['R']
        if model_name in ['sdt', 'uvsdt', 'dpsdt']:
            d = model_fit[model_name]['non_c_vars']['d']

        if model_name == 'ht':
            x, y = high_threshold(R=R)
        elif model_name == 'sdt':
            x, y = signal_detection(d=d)
        elif model_name == 'uvsdt':
            signal_variance = model_fit[model_name]['non_c_vars']['signal_var']
            x, y = signal_detection(d=d, signal_var=signal_variance)
        elif model_name == 'dpsdt':
            x, y = dual_process(R=R, d=d)  # Using the fitted R and d parameters
            # Get the familiarity
            F = strength_estimate(d=d, c=c_vars)

        param_msg = f"R = {R:.2f}\nF = {F:.2f}" if model_name == 'dpsdt' else ""

        # Plot -------------------------------------------#
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        # Chance line
        ax.plot([0, 1], [0, 1], ls='dashed', c='grey')
        # ROC data
        ax.plot(roc_noise, roc_signal, marker='o', label='data')
        # Modelled data
        ax.plot(x, y, label='dual process')
        # More information
        ax.text(x=2 / 3, y=1 / 3, s=f"G\N{SUPERSCRIPT TWO}{sumg2:.2f}\n{param_msg}")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right')
        ax.set_title(f'ROC\n({model_name.upper()} model)')

        plt.show()
