import numpy as np
import scipy
from scipy import integrate
import json
import argparse
import matplotlib.pyplot as plt
import os
import xarray as xr
import pickle
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

def xr_summary(data_array, sel=dict(), timeslice=slice(0, None), sum_over=['node']):
    """from Ethan
    """
    isel = {'time': timeslice}
    da = data_array[isel].loc[sel].sum(dim=sum_over)
    return da

def deterministic_pars(param):

    return param

def make_array(dims, val):

    new_arr = np.zeros(dims)
    new_arr.fill(val)

    return new_arr

# ----------------------------------------------------------------------------------------------
# -- define the system of differential equations
# ----------------------------------------------------------------------------------------------

class SEIR:

    def __init__(self, paramset):

        # set dimensions for storage arrays as time, nodes, ages based on these values
        self.days = paramset['days']
        self.interval_per_day = 10
        self.duration = self.days * self.interval_per_day  # number of discrete time steps with 10 time steps per day
        self.n_age = paramset['n_age']
        self.nodes = paramset['n_nodes']
        self.stochastic = paramset['stochastic']

        # set stochastic parameters, if applicable
        if self.stochastic == "True":
            self.beta = abs(np.random.normal(paramset['beta'], 0.5))
            self.sigma = abs(np.random.normal(paramset['sigma'] / self.interval_per_day, 0.2))  # rate E -> I
        else:
            self.beta = paramset['beta']  # transmission rate
            self.sigma = paramset['sigma'] / self.interval_per_day  # rate E -> I
        if self.stochastic == "True":
            self.distr_function = np.random.poisson
        else:
            self.distr_function = deterministic_pars

        # set other epi parameters
        self.mu = paramset['mu'] / self.interval_per_day  # death rate from infection
        self.gamma = paramset['gamma'] / self.interval_per_day  # recovery rate
        self.omega = paramset['omega']  # relative infectiousness
        self.phi = np.array(paramset['phi'])  # contact matrix for school; set aside now
        self.outpath = paramset['outpath']
        self.sim_idx = paramset['sim_idx']
        self.force_inf = 0
        self.rate_s2e = 0  # same as force of infection

        # set initial population conditions
        self.Sus = make_array(dims=(self.duration, self.nodes, self.n_age), val=0)
        self.Sus[0] = paramset['start_S']
        self.Exp = make_array(dims=(self.duration, self.nodes, self.n_age), val=0)
        self.Exp[0] = paramset['start_E']
        self.Inf = make_array(dims=(self.duration, self.nodes, self.n_age), val=0)
        self.Inf[0] = paramset['start_I']
        self.Rec = make_array(dims=(self.duration, self.nodes, self.n_age), val=0)
        self.Rec[0] = paramset['start_R']
        self.N = make_array(dims=(self.duration, self.nodes, self.n_age), val=0)
        self.N[0] = np.array(paramset['start_S']) + np.array(paramset['start_E']) + np.array(paramset['start_I']) + np.array(paramset['start_R'])

    def N_t(self, t, a, n):

        self.N[t][n][a] = self.Sus[t][n][a] + self.Exp[t][n][a] + self.Inf[t][n][a] + self.Rec[t][n][a]

    def calc_force_infection(self, phi, t, a, n):

        self.force_inf += ((self.beta * phi * self.omega * self.Sus[t-1][n][a] * self.Inf[t-1][n][a]) / self.N[t-1][n][a])
        return self.distr_function(self.force_inf)

    def calc_exp_to_inf(self, t, a, n):

        return self.distr_function(self.sigma * self.Exp[t-1][n][a])

    def calc_inf_to_rec(self, t, a, n):

        return self.distr_function(self.gamma * self.Inf[t-1][n][a])

    def seir(self):

        for t in range(1, self.duration):

            for n1 in range(self.nodes):

                for n2 in range(self.nodes):

                    for a1 in range(self.n_age):

                        for a2 in range(self.n_age):

                            ## set rates
                            # force of infection from a2 on to a1
                            try:
                                rate_s2e = self.calc_force_infection(phi=self.phi[n1, n2, a1, a2], t=t, a=a2, n=n1)
                            except IndexError:
                                print('breakpoint')
                                #breakpoint()

                            # other transition transition rates for a1
                            rate_e2i = self.calc_exp_to_inf(t, a1, n1)
                            rate_i2r = self.calc_inf_to_rec(t, a1, n1)

                            ## SUSCEPTIBLE
                            self.Sus[t][n1][a1] = max(0, self.Sus[t-1][n1][a1] - rate_s2e)
                            if not self.Sus[t][n1][a1] > 0:
                                rate_s2e = self.Sus[t - 1][n1][a1]

                            ## EXPOSED
                            self.Exp[t][n1][a1] = max(0, self.Exp[t-1][n1][a1] + rate_s2e - rate_e2i)
                            if not self.Exp[t][n1][a1] > 0:
                                rate_e2i = self.Exp[t - 1][n1][a1] + rate_s2e

                            ## INFECTED
                            self.Inf[t][n1][a1] = max(0, self.Inf[t-1][n1][a1] + rate_e2i - rate_i2r)
                            if not self.Inf[t][n1][a1] > 0:
                                rate_i2r = self.Inf[t - 1][n1][a1] + rate_e2i

                            ## RECOVERED
                            self.Rec[t][n1][a1] = max(0, self.Rec[t-1][n1][a1] + rate_i2r)

                            self.N_t(t, a1, n1)  # update population totals

                        # turn each compartment into a data array
                        s_da = xr.DataArray(self.Sus, dims=['time', 'node', 'age'],
                                            coords={'time': [i for i in range(self.duration)], 'age': ['young', 'old'],
                                                    'node': [i for i in range(self.nodes)]})
                        e_da = xr.DataArray(self.Exp, dims=['time', 'node', 'age'],
                                            coords={'time': [i for i in range(self.duration)], 'age': ['young', 'old'],
                                                    'node': [i for i in range(self.nodes)]})
                        i_da = xr.DataArray(self.Inf, dims=['time', 'node', 'age'],
                                            coords={'time': [i for i in range(self.duration)], 'age': ['young', 'old'],
                                                    'node': [i for i in range(self.nodes)]})
                        r_da = xr.DataArray(self.Rec, dims=['time', 'node', 'age'],
                                            coords={'time': [i for i in range(self.duration)], 'age': ['young', 'old'],
                                                    'node': [i for i in range(self.nodes)]})

                        # align all compartment data arrays into a single dataset
                        self.final = xr.Dataset(
                            data_vars={
                                'S': (('time', 'node', 'age'), s_da),
                                'E': (('time', 'node', 'age'), e_da),
                                'I': (('time', 'node', 'age'), i_da),
                                'R': (('time', 'node', 'age'), r_da)
                            },
                            coords={'time': [i for i in range(self.duration)], 'age': ['young', 'old'], 'node': [i for i in range(self.nodes)]}
                        )

    def save_timeseries(self):

        save_name = '{}.pckl'.format(self.outpath)
        with open(save_name, 'w') as f:
            pickle.dump(self.final, f)

    def plot_timeseries(self):

        final_s = xr_summary(self.final.S, sel={'age': 'young'}, timeslice=slice(0, self.duration), sum_over='node')
        final_e = xr_summary(self.final.E, sel={'age': 'young'}, timeslice=slice(0, self.duration))
        final_i = xr_summary(self.final.I, sel={'age': 'young'}, timeslice=slice(0, self.duration))
        final_r = xr_summary(self.final.R, sel={'age': 'young'}, timeslice=slice(0, self.duration))

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        final_s.plot(ax=ax, color='b')
        final_e.plot(ax=ax, color='g')
        final_i.plot(ax=ax, color='r')
        final_r.plot(ax=ax, color='k')

        plt.legend(("S", "E", "I", "R"), loc=0)
        plt.ylabel("Population Size")
        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.title("SEIR Model")
        plt.savefig('{}.png'.format(self.outpath))
        plt.tight_layout()

        plt.show()

def main(opts):

    # ----- Load and validate parameters -----#

    pars = acquire_params(opts.paramfile)

    float_keys = ['beta', 'mu', 'sigma', 'gamma', 'omega']
    int_keys = ['days'] # 'start_S', 'start_E', 'start_I', 'start_R', (initial conditions now arrays)
    str_keys = ['outpath']
    # not checked: phi
    validate_params(pars, float_keys, int_keys, str_keys)

    # ----- Run model if inputs are valid -----#

    n_sims = pars.pop('n_sims')

    if n_sims > 1:
        raise NotImplementedError('Support for saving multiple simulations output not yet implemented.')

    while n_sims > 0:
        pars['sim_idx'] = n_sims
        seir_model = SEIR(pars)
        seir_model.seir()
        #seir_model.save_timeseries()
        seir_model.plot_timeseries()
        n_sims -= 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paramfile', help='Path to parameters file.')

    opts = parser.parse_args()

    main(opts)


