import numpy as np
from numpy.random import exponential
import scipy
from scipy import integrate
import json
import argparse
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------------------------------------
# -- Read in parameters file
# ----------------------------------------------------------------------------------------------

def acquire_params(filename):

    with open(filename, 'r') as param_file:
        params = json.loads(param_file.read())

    return params

# ----------------------------------------------------------------------------------------------
# -- Validate Inputs
# ----------------------------------------------------------------------------------------------

def validate_params(param_dict, float_keys, int_keys, str_keys):

    all_keys = float_keys + int_keys + str_keys
    for key in all_keys:
        if key not in param_dict.keys():
            raise ValueError('Parameter {} missing from input file.'.format(key))

    for key in float_keys:
        if type(param_dict[key]) != float:
            raise ValueError('Parameter {} is not specified as a float.'.format(key))

    for key in int_keys:
        if type(param_dict[key]) != int:
            raise ValueError('Parameter {} is not specified as an integer.'.format(key))

    for key in str_keys:
        if type(param_dict[key]) != str:
            raise ValueError('Parameter {} is not specified as a string.'.format(key))


# ----------------------------------------------------------------------------------------------
# -- define the system of differential equations
# ----------------------------------------------------------------------------------------------

class SEIR:

    def __init__(self, beta_lambda, mu, sigma, gamma, omega, start_S, start_E, start_I, start_R, duration, outdir, n_runs):

        self.beta_lambda = beta_lambda
        self.beta = self.draw_beta()
        self.mu = mu  # death rate from infection
        self.sigma = sigma  # rate E -> I
        self.gamma = gamma  # recovery rate
        self.omega = omega  # waning immunity
        self.start_S = start_S
        self.start_E = start_E
        self.start_I = start_I
        self.start_R = start_R
        self.duration = duration
        self.outdir = outdir
        self.n_runs = n_runs
        self.R = [self.start_S, self.start_E, self.start_I, self.start_R]

    def draw_beta(self):
        return exponential(self.beta_lambda)

    def seir(self, x, t):

        S = x[0]
        E = x[1]
        I = x[2]
        R = x[3]

        y = np.zeros(4)

        y[0] = self.mu - ((self.beta * I) + self.mu) * S + (self.omega * R)
        y[1] = (self.beta * S * I) - (self.mu + self.sigma) * E
        y[2] = (self.sigma * E) - (self.mu + self.gamma) * I
        y[3] = (self.gamma * I) - (self.mu * R) - (self.omega * R)

        return y

    def integrate(self):

        time = np.arange(0, self.duration, 0.01)
        results = scipy.integrate.odeint(self.seir, self.R, time)

        return results

def plot_timeseries(results, savedir):

    time = np.arange(0, len(results[1]))

    plt.figure(figsize=(5,8), dpi=300)

    for r in results:
        plt.plot(
            time, r[:, 2], "r"
        )
    plt.ylabel("Population Size")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("SEIR Model | Infection Time Series, Stochastic Beta")
    plt.savefig(os.path.join(savedir, 'Stochastic_SEIR_Model.png'))
    plt.show()

def main(opts):

    # ----- Load and validate parameters -----#

    pars = acquire_params(opts.paramfile)

    float_keys = ['beta_lambda', 'mu', 'sigma', 'gamma']
    int_keys = ['start_S', 'start_E', 'start_I', 'start_R', 'duration', 'n_runs']
    str_keys = ['outdir']
    validate_params(pars, float_keys, int_keys, str_keys)

    # ----- Run model if inputs are valid -----#

    results = []
    for i in range(pars['n_runs']):
        seir_model = SEIR(**pars)
        results.append(seir_model.integrate())

    plot_timeseries(results, pars['outdir'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paramfile', help='Path to parameters file.')

    opts = parser.parse_args()

    main(opts)


