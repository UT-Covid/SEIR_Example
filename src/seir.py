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

def discrete_time_approx(rate, timestep):
    """

    :param rate: daily rate
    :param timestep: timesteps per day
    :return: rate rescaled by time step
    """
    if timestep == 1:
        return rate
    else:
        return (1 - (1 - rate)**(1/timestep))


# ----------------------------------------------------------------------------------------------
# -- define the system of difference equations
# ----------------------------------------------------------------------------------------------

class SEIR:

    def __init__(self, paramset):

        # set numpy warning, error handling
        np.seterr(all='raise')

        # set dimensions for storage arrays as time, nodes, ages based on these values
        self.days = paramset['days']
        self.interval_per_day = paramset['interval_per_day']
        self.duration = self.days * self.interval_per_day
        self.n_age = paramset['n_age']
        self.nodes = paramset['n_nodes']
        self.stochastic = paramset['stochastic']

        ## RATE PARAMETERS
        # set stochastic parameters, if applicable
        if self.stochastic == "True":
            self.beta = abs(np.random.normal(paramset['beta'], 0.5))
            self.sigma = abs(np.random.normal(
                discrete_time_approx(rate=['sigma'], timestep=self.interval_per_day),
                0.2))  # rate E -> I
        else:
            self.beta = paramset['beta']  # transmission rate
            self.sigma = discrete_time_approx(
                rate=paramset['sigma'],
                timestep=self.interval_per_day
            )# rate E -> I
        self.mu = discrete_time_approx(
            rate=paramset['mu'],
            timestep=self.interval_per_day
        )  # death rate from infection
        self.gamma = discrete_time_approx(
            rate=paramset['gamma'],
            timestep=self.interval_per_day
        )  # recovery rate

        # OTHER EPI AND SIMULATION PARAMETERS
        self.omega = paramset['omega']  # relative infectiousness
        self.phi = np.array(paramset['phi'])
        self.outpath = paramset['outpath']
        self.sim_idx = paramset['sim_idx']
        self.precalc_partial_foi()  # pre-multiply omega, beta, and each element in phi to reduce number of ops in for-loop

        ## TRANSITION PROBABILITY
        if self.stochastic == "True":
            self.distr_function = np.random.poisson
        else:
            self.distr_function = deterministic_pars

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

    def precalc_partial_foi(self):

        beta_omega = self.omega * self.beta
        partial_foi = np.zeros([self.nodes, self.nodes, self.n_age, self.n_age])
        for i in  range(self.nodes):
            for j in range(self.nodes):
                for k in range(self.n_age):
                    for m in range(self.n_age):
                        partial_foi[i, j, k, m] = beta_omega * self.phi[i, j, k, m]

        self.partial_foi = partial_foi

    def N_t(self, t, a, n):

        self.N[t][n][a] = self.Sus[t][n][a] + self.Exp[t][n][a] + self.Inf[t][n][a] + self.Rec[t][n][a]

    def calc_force_infection(self, t, a1, a2, n1, n2):

        # if the node, age subpopulation is 0, no addition to overall force of infection
        if self.N[t-1][n2][a2] < 1e-12:
            return 0
        else:
            foi = (self.partial_foi[n1, n2, a1, a2] * self.Sus[t - 1][n1][a1] * self.Inf[t - 1][n2][a2]) / self.N[t - 1][n2][a2]
            #foi = (self.beta * phi * self.omega * self.Sus[t-1][n][a] * self.Inf[t-1][n][a]) / self.N[t-1][n][a]
            return self.distr_function(foi)

    def calc_exp_to_inf(self, t, a, n):

        return self.distr_function(self.sigma * self.Exp[t-1][n][a])

    def calc_inf_to_rec(self, t, a, n):

        return self.distr_function(self.gamma * self.Inf[t-1][n][a])

    def seir(self):

        for t in range(1, self.duration):

            for n1 in range(self.nodes):

                for a1 in range(self.n_age):

                    # force of infection on node i, age i from all other nodes, ages
                    rate_s2e = 0

                    for n2 in range(self.nodes):

                        for a2 in range(self.n_age):

                            try:
                                rate_s2e += self.calc_force_infection(t=t, a1=a1, a2=a2, n1=n1, n2=n2)
                                # force of infection from a2 on to a1
                            except FloatingPointError:
                                # custom exception msg to help pinpoint problem
                                print(self.beta, self.phi[n1, n2, a1, a2], self.omega, self.Sus[t-1][n1][a1], self.Inf[t-1][n2][a2], self.N[t-1][n2][a2])
                                raise FloatingPointError('iteration {}'.format(t))

                    # other transition transition rates for a1, not dependent on interaction with other nodes, ages
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


