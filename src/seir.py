import numpy as np
import scipy
from scipy import integrate
import json
import argparse
import matplotlib.pyplot as plt
import os
import datetime as dt

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
# -- static methods
# ----------------------------------------------------------------------------------------------

def deterministic_pars(param):

    return param

# ----------------------------------------------------------------------------------------------
# -- define the system of differential equations
# ----------------------------------------------------------------------------------------------

class SEIR:

    def __init__(self, days, beta, mu, sigma, gamma, omega, start_S, start_E, start_I, start_R, outdir):

        self.days = days
        self.interval_per_day = 10
        self.duration = self.days * self.interval_per_day  # number of discrete time steps with 10 time steps per day
        self.beta = beta  # transmission rate
        self.mu = mu / self.interval_per_day  # death rate from infection
        self.sigma = sigma / self.interval_per_day  # rate E -> I
        self.gamma = gamma / self.interval_per_day  # recovery rate
        self.omega = omega  # relative infectiousness
        self.phi = 1  # contact matrix for school; set aside now
        self.start_S = start_S
        self.start_E = start_E
        self.start_I = start_I
        self.start_R = start_R
        self.outdir = outdir
        self.Sus = [self.start_S]  # speed up code by setting array size at the beginning
        self.Exp = [self.start_E]
        self.Inf = [self.start_I]
        self.Rec = [self.start_R]
        self.N = [self.Sus[0] + self.Exp[0] + self.Inf[0] + self.Rec[0]]
        self.stochastic = False
        self.force_inf = 0
        self.rate_s2e = 0  # force of infection
        self.distr_function = deterministic_pars
        if self.stochastic:
            self.distr_function = np.random.poisson

    def N_t(self, t):

        self.N.append(self.Sus[t] + self.Exp[t] + self.Inf[t] + self.Rec[t])

    def calc_force_infection(self, t):

        self.force_inf += ((self.beta * self.phi * self.omega * self.Sus[t-1] * self.Inf[t-1]) / self.N[t-1])
        return self.distr_function(self.force_inf)

    def calc_exp_to_inf(self, t):

        return self.distr_function(self.sigma * self.Exp[t-1])

    def calc_inf_to_rec(self, t):

        return self.distr_function(self.gamma * self.Inf[t-1])

    def seir(self):

        for t in range(1, self.duration):

            ## set rates
            rate_s2e = self.calc_force_infection(t)
            rate_e2i = self.calc_exp_to_inf(t)
            rate_i2r = self.calc_inf_to_rec(t)

            self.Sus.append(max(0, self.Sus[t-1] - rate_s2e))
            if not self.Sus[t] > 0:
                rate_s2e = self.Sus[t-1]
            self.Exp.append(max(0, self.Exp[t-1] + rate_s2e - rate_e2i))
            if not self.Exp[t] > 0:
                rate_e2i = self.Exp[t-1] + rate_s2e
            self.Inf.append(max(0, self.Inf[t-1] + rate_e2i - rate_i2r))
            if not self.Inf[t] > 0:
                rate_i2r = self.Inf[t-1] + rate_e2i
            self.Rec.append(max(0, self.Rec[t-1] + rate_i2r))

            self.N_t(t)  # update population totals

    def plot_timeseries(self):

        time = np.arange(0, self.duration)

        plt.figure(figsize=(5, 8), dpi=300)

        plt.plot(
            time, self.Sus, "k",
            time, self.Exp, "g",
            time, self.Inf, "r",
            time, self.Rec, "b",)
        plt.legend(("S", "E", "I", "R"), loc=0)
        plt.ylabel("Population Size")
        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.title("SEIR Model")
        plt.savefig(os.path.join(self.outdir, 'SEIR_Model.png'))
        plt.show()

def main(opts):

    # ----- Load and validate parameters -----#

    pars = acquire_params(opts.paramfile)

    float_keys = ['beta', 'mu', 'sigma', 'gamma', 'omega']
    int_keys = ['start_S', 'start_E', 'start_I', 'start_R', 'days']
    str_keys = ['outdir']
    validate_params(pars, float_keys, int_keys, str_keys)

    # ----- Run model if inputs are valid -----#

    seir_model = SEIR(**pars)
    seir_model.seir()
    seir_model.plot_timeseries()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paramfile', help='Path to parameters file.')

    opts = parser.parse_args()

    main(opts)


