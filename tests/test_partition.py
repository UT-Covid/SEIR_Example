import sys
import pytest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray
from src.seir import *
from src.implicit_node import *
from copy import deepcopy

SHOW_PLT = False


def update_start_pop(travel_df):
    grouped = travel_df.groupby(['source', 'age_src'])['n'].sum().reset_index()
    nodes = sorted(travel_df['source'].unique())
    ages = sorted(travel_df['age_src'].unique(), reverse=True)
    pop_arr_s = np.zeros([len(nodes), len(ages)])
    pop_arr_e = np.zeros([len(nodes), len(ages)])
    pop_arr_i = np.zeros([len(nodes), len(ages)])
    pop_arr_r = np.zeros([len(nodes), len(ages)])
    for i, node in enumerate(nodes):
        for j, age in enumerate(ages):
            new_total = grouped[(grouped['source']==node) & (grouped['age_src']==age)]['n'].item()
            if new_total > 2:
                pop_arr_s[i, j] = new_total-1
                pop_arr_i[i, j] = 1
    return pop_arr_s, pop_arr_e, pop_arr_i, pop_arr_r


def update_travel(travel_df, new_count, source, destination, age_src='young', age_dest='young'):
    travel_idx = travel_df.set_index(['source', 'destination', 'age_src', 'age_dest'])
    travel_dict = travel_idx.to_dict()
    travel_dict['n'][(source, destination, age_src, age_dest)] = new_count
    updated_df = pd.DataFrame.from_dict(travel_dict).reset_index()
    updated_df.columns = ['source', 'destination', 'age_src', 'age_dest', 'destination_type', 'n']
    return updated_df


def update_contact_rate(contact_df, new_rate, age1='young', age2='young'):
    contact_idx = contact_df.set_index(['age1', 'age2'])
    contact_dict = contact_idx.to_dict()
    contact_dict['daily_per_capita_contacts'][(age1, age2)] = new_rate
    updated_df = pd.DataFrame.from_dict(contact_dict).reset_index()
    updated_df.columns = ['age1', 'age2', 'daily_per_capita_contacts']
    return updated_df


def discrete_time_approx(rate, timestep):
    """
    :param rate: daily rate
    :param timestep: timesteps per day
    :return: rate rescaled by time step
    """
    return (1 - (1 - rate)**(1/timestep))


@pytest.fixture
def atol():
    """ #0.001 max persons difference"""
    return 0.001


@pytest.fixture
def rtol():
    """ #0.05% max difference"""
    return 0.0005


@pytest.fixture
def params_template():
    return {
      "mu": 0.0,
      "sigma": 0.5,
      "beta": 0.1,
      "gamma": 0.2,
      "omega": 0.1,
      "start_S": [[24, 0], [49, 0]],
      "start_E": [[0, 0], [0, 0]],
      "start_I": [[1, 0], [1, 0]],
      "start_R": [[0, 0], [0, 0]],
      "days": 30,
      "outpath": "outputs/multiple_nodes",
      "phi": [], # fill in after partitioning
      "n_sims": 1,
      "stochastic": "False",
      "n_age": 2,
      "n_nodes": 2,
      "sim_idx": 0, # single deterministic run
      "interval_per_day": 10
    }


@pytest.fixture
def travel():
    return pd.read_csv('inputs/travel2.csv')


@pytest.fixture
def contact():
    return pd.read_csv('inputs/contact.csv')


@pytest.fixture
def partition(travel, contact):
    return partition_contacts(travel, contact, daily_timesteps=10)


@pytest.fixture
def phi_matrix(partition):
    return contact_matrix(partition)


@pytest.fixture
def test1(params_template, phi_matrix):
    test1 = deepcopy(params_template)
    test1['phi'] = phi_matrix
    return test1


@pytest.fixture
def ref_params():
    return {
      "mu": 0.0,
      "sigma": 0.5,
      "beta": 0.1,
      "gamma": 0.2,
      "omega": 0.1,
      "start_S": [[73, 0]],
      "start_E": [[0, 0]],
      "start_I": [[2, 0]],
      "start_R": [[0, 0]],
      "days": 30,
      "outpath": "outputs/single_node",
      "phi": [[[[5/10, 0], [0, 0]]]],
      "n_sims": 1,
      "stochastic": "False",
      "n_age": 2,
      "n_nodes": 1,
      "sim_idx": 0,
      "interval_per_day": 10
    }


@pytest.fixture
def polymod():
    return pd.read_csv('./data/Cities_Data/ContactMatrixAll_5AgeGroups.csv',
                        header=None)


@pytest.fixture
def max_cr(polymod):
    return polymod.max().max()


@pytest.fixture
def min_cr(polymod):
    return polymod.min().min()


@pytest.fixture
def max_test(max_cr, contact, travel, params_template):
    max_contact = update_contact_rate(contact, max_cr)
    max_partition = partition_contacts(travel, max_contact, daily_timesteps=10)
    max_phi_matrix = contact_matrix(max_partition)

    test_max = deepcopy(params_template)
    test_max['phi'] = max_phi_matrix
    return test_max
    # test_max_model = SEIR(test_max)
    # test_max_model.seir()


@pytest.fixture
def max_ref(max_cr, ref_params):
    ref_max = deepcopy(ref_params)
    ref_max['phi'] = [[[[max_cr/10, 0], [0, 0]]]]
    return ref_max
    # ref_max_model = SEIR(ref_max)
    # ref_max_model.seir()


@pytest.fixture
def min_contact(contact, min_cr):
    return update_contact_rate(contact, min_cr)


@pytest.fixture
def min_test(params_template, travel, min_contact):
    min_partition = partition_contacts(travel, min_contact, daily_timesteps=10)
    min_phi_matrix = contact_matrix(min_partition)
    test_min = deepcopy(params_template)
    test_min['phi'] = min_phi_matrix
    return test_min


@pytest.fixture
def ref_min(min_cr, ref_params):
    d = deepcopy(ref_params)
    d['phi'] = [[[[min_cr/10, 0], [0, 0]]]]
    return d


@pytest.fixture
def travel3():
    """# # Sensitivity analysis: additional nodes"""
    return pd.read_csv('inputs/travel3.csv')


@pytest.fixture
def test3(travel, travel3, params_template, contact):
    new_travel = update_travel(travel, 50, source='A', destination='A')
    pop_s, pop_e, pop_i, pop_r = update_start_pop(new_travel)

    partition3 = partition_contacts(travel3, contact, daily_timesteps=10)
    phi_matrix3 = contact_matrix(partition3)
    pop_s, pop_e, pop_i, pop_r = update_start_pop(travel3)
    test3_params = deepcopy(params_template)
    test3_params['phi'] = phi_matrix3
    test3_params['start_S'] = pop_s
    test3_params['start_E'] = pop_e
    test3_params['start_I'] = pop_i
    test3_params['start_R'] = pop_r
    test3_params['n_nodes'] = 3
    return test3_params


@pytest.fixture
def ref3(params_template, travel3):
    ref3_params = deepcopy(params_template)
    ref3_params['phi'] = [[[[5/10, 0], [0, 0]]]]
    ref3_params_s = np.array([sorted(travel3.groupby(['age_src'])['n'].sum(), reverse=True)])
    ref3_params['start_S'] = ref3_params_s[0, 0] - 3
    ref3_params['start_E'] = np.array([[0, 0]])
    ref3_params['start_I'] = np.array([[3, 0]])
    ref3_params['start_R'] = np.array([[0, 0]])
    ref3_params['n_nodes'] = 1
    return ref3_params


@pytest.fixture
def test3_min(min_contact, travel3, ref3, test3):
    partition3_min = partition_contacts(travel3, min_contact, daily_timesteps=10)
    phi_matrix3_min = contact_matrix(partition3_min)
    test3_min_params = deepcopy(test3)
    test3_min_params['phi'] = phi_matrix3_min
    return test3_min_params


@pytest.fixture
def ref3_min(min_cr, ref3):
    ref3_min_params = deepcopy(ref3)
    ref3_min_params['phi'] = [[[[min_cr/10, 0], [0, 0]]]]
    return ref3_min_params


@pytest.fixture(params=[
    ('test1', 'ref_params'),
    ('max_test', 'max_ref'),
    ('min_test', 'ref_min'),
    ('test3', 'ref3'),
    ('test3_min', 'ref3_min'),
])
def comparison(request):
    """A workaround alternative to passing fixtures in pytest.mark.parametrize"""
    return [request.getfixturevalue(f) for f in request.param]


# the only actual test function in this module
def test_partition(comparison, atol, rtol):
    # comparison is a tuple of resolved fixture values. expand it
    test, ref = comparison
    kwargs = dict(atol=atol, rtol=rtol)

    # always use allclose
    method = xarray.testing.assert_allclose

    # generate and run models
    test_model = SEIR(test)
    ref_model = SEIR(ref)
    test_model.seir()
    ref_model.seir()

    test_model_s = xr_summary(test_model.final.S, sel={'age': 'young'}, timeslice=slice(0, test_model.duration), sum_over='node')
    ref_model_s = xr_summary(ref_model.final.S, sel={'age': 'young'}, timeslice=slice(0, ref_model.duration), sum_over='node')
    diff_s = test_model_s - ref_model_s

    test_model_e = xr_summary(test_model.final.E, sel={'age': 'young'}, timeslice=slice(0, test_model.duration), sum_over='node')
    ref_model_e = xr_summary(ref_model.final.E, sel={'age': 'young'}, timeslice=slice(0, ref_model.duration), sum_over='node')
    diff_e = test_model_e - ref_model_e

    test_model_i = xr_summary(test_model.final.I, sel={'age': 'young'}, timeslice=slice(0, test_model.duration), sum_over='node')
    ref_model_i = xr_summary(ref_model.final.I, sel={'age': 'young'}, timeslice=slice(0, ref_model.duration), sum_over='node')
    diff_i = test_model_i - ref_model_i

    test_model_r = xr_summary(test_model.final.R, sel={'age': 'young'}, timeslice=slice(0, test_model.duration), sum_over='node')
    ref_model_r = xr_summary(ref_model.final.R, sel={'age': 'young'}, timeslice=slice(0, ref_model.duration), sum_over='node')
    diff_r = test_model_r - ref_model_r

    try:
        method(test_model_s, ref_model_s, **kwargs)
    except AssertionError as _err:
        print('Differing values for susceptible timeseries.')
        raise

    try:
        method(test_model_e, ref_model_e, **kwargs)
    except AssertionError as _err:
        print('Differing values for exposed timeseries.')
        raise

    try:
        method(test_model_i, ref_model_i, **kwargs)
    except AssertionError as _err:
        print('Differing values for infected timeseries.')
        raise

    try:
        method(test_model_r, ref_model_r, **kwargs)
    except AssertionError as _err:
        print('Differing values for recovered timeseries.')
        raise

    return diff_s, diff_e, diff_i, diff_r



# plot test1 vs ref_model
def plt_test1_vs_ref_params():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    s_diff.plot(ax=ax, color='b')
    e_diff.plot(ax=ax, color='g')
    i_diff.plot(ax=ax, color='r')
    r_diff.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("N Partition - N Baseline")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Two Local, One Contextual vs Baseline Mixing = 5 contacts/person/day")
    plt.tight_layout()
    #ax.set_ylim(-2, 3)
    plt.axhline(y=0, c='gray', ls='dotted')
    if SHOW_PLT is True:
        plt.show()


# strict 1
# test_partition(test1_model, ref_model, method=xarray.testing.assert_equal)
# update_contact_rate(contact, 0.3)

# test_max_model vs ref_max_model
# ds_max, de_max, di_max, dr_max = test_partition(test_max_model, ref_max_model, atol=abs_tol, rtol=rel_tol)


# plot
def plt_max_test_vs_ref_max():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ds_max.plot(ax=ax, color='b')
    de_max.plot(ax=ax, color='g')
    di_max.plot(ax=ax, color='r')
    dr_max.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("N Partition - N Baseline")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Two Local, One Contextual vs Baseline Mixing = 10.2 contacts/person/day")
    plt.tight_layout()
    #ax.set_ylim(-2, 3)
    plt.axhline(y=0, c='gray', ls='dotted')
    if SHOW_PLT is True:
        plt.show()

# test min
# ds_min, de_min, di_min, dr_min = test_partition(test_min_model, ref_min_model, atol=abs_tol, rtol=rel_tol)


def plt_min_test_vs_ref_min():
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ds_min.plot(ax=ax, color='b')
    de_min.plot(ax=ax, color='g')
    di_min.plot(ax=ax, color='r')
    dr_min.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("N Partition - N Baseline")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Two Local, One Contextual vs Baseline Mixing = 0.2 contacts/person/day")
    plt.tight_layout()
    #ax.set_ylim(-2,3)
    plt.axhline(y=0, c='gray', ls='dotted')
    if SHOW_PLT is True:
        plt.show()

#
# ref_min_model.plot_timeseries()
# test_min_model.plot_timeseries()
# ref_max_model.plot_timeseries()
# test_max_model.plot_timeseries()


# # Sensitivity analysis: population size
@pytest.fixture
def pop_template():
    return {
      "mu": 0.0,
      "sigma": 0.5,
      "beta": 0.1,
      "gamma": 0.2,
      "omega": 0.1,
      "start_S": [[24, 0], [49, 0]],
      "start_E": [[0, 0], [0, 0]],
      "start_I": [[1, 0], [1, 0]],
      "start_R": [[0, 0], [0, 0]],
      "days": 30,
      "outpath": "outputs/multiple_nodes",
      "phi": [], # fill in after partitioning
      "n_sims": 1,
      "stochastic": "False",
      "n_age": 2,
      "n_nodes": 2,
      "sim_idx": 0, # single deterministic run
      "interval_per_day": 10
    }

# test model 3
# s3_diff, e3_diff, i3_diff, r3_diff = test_partition(test3_model, ref3_model, atol=abs_tol, rtol=rel_tol)

# plot
def plt_test3_vs_ref3():
    test3_model.plot_timeseries()
    ref3_model.plot_timeseries()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    s3_diff.plot(ax=ax, color='b')
    e3_diff.plot(ax=ax, color='g')
    i3_diff.plot(ax=ax, color='r')
    r3_diff.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("N Partition - N Baseline")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Three Local, One Contextual vs Baseline Mixing = 5 contacts/person/day")
    plt.tight_layout()
    #ax.set_ylim(-2, 3)
    plt.axhline(y=0, c='gray', ls='dotted')
    # plt.show()



def plot_local_3v2():
    local3v2_contact5_s = s3_diff - s_diff
    local3v2_contact5_e = e3_diff - e_diff
    local3v2_contact5_i = i3_diff - i_diff
    local3v2_contact5_r = r3_diff - r_diff

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    local3v2_contact5_s.plot(ax=ax, color='b')
    local3v2_contact5_e.plot(ax=ax, color='g')
    local3v2_contact5_i.plot(ax=ax, color='r')
    local3v2_contact5_r.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("Population Size")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Three Local, One Contextual vs Two Local, One Contextual")
    plt.tight_layout()
    # plt.show()

    ref3_model.plot_timeseries()
    test3_model.plot_timeseries()


# test3_min vs ref3_min
# s3_diff_min, e3_diff_min, i3_diff_min, r3_diff_min = test_partition(test3_min_model, ref3_min_model, atol=abs_tol, rtol=rel_tol)

#
# # plot
# fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# s3_diff_min.plot(ax=ax, color='b')
# e3_diff_min.plot(ax=ax, color='g')
# i3_diff_min.plot(ax=ax, color='r')
# r3_diff_min.plot(ax=ax, color='k')
# plt.legend(("S", "E", "I", "R"), loc=0)
# plt.ylabel("N Partition - N Baseline")
# plt.xlabel("Time")
# plt.xticks(rotation=45)
# plt.title("Three Local, One Contextual vs Baseline Mixing = 0.2 contacts/person/day")
# plt.tight_layout()
# plt.axhline(y=0, c='gray', ls='dotted')
# #ax.set_ylim(-2, 3)
# # plt.show()
#
#
# partition3_max = partition_contacts(travel3, max_contact, daily_timesteps=10)
# phi_matrix3_max = contact_matrix(partition3_max)
#
# test3_max = deepcopy(test3)
# test3_max['phi'] = phi_matrix3_max
#
# ref3_max = deepcopy(ref3)
# ref3_max['phi'] = [[[[max_cr/10, 0], [0, 0]]]]
#
# test3_max_model = SEIR(test3_max)
# test3_max_model.seir()
#
# ref3_max_model = SEIR(ref3_max)
# ref3_max_model.seir()
#
#
# # test
# s3_diff_max, e3_diff_max, i3_diff_max, r3_diff_max = test_partition(test3_max_model, ref3_max_model, atol=abs_tol, rtol=rel_tol)
#
# # plot
# fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# s3_diff_max.plot(ax=ax, color='b')
# e3_diff_max.plot(ax=ax, color='g')
# i3_diff_max.plot(ax=ax, color='r')
# r3_diff_max.plot(ax=ax, color='k')
# plt.legend(("S", "E", "I", "R"), loc=0)
# plt.xlabel("Time")
# plt.xticks(rotation=45)
# plt.title("Three Local, One Contextual vs Baseline Mixing = 10.2 contacts/person/day")
# plt.tight_layout()
# plt.ylabel("N Partition - N Baseline")
# plt.axhline(y=0, c='gray', ls='dotted')
# #ax.set_ylim(-2, 3)
# # plt.show()
#
#
#
# def sixteen_nodes():
#     raise NotImplementedError()
#     # # 16 Nodes
#
#     travel16 = pd.read_csv('inputs/travel16.csv')
#     partition16 = partition_contacts(travel16, contact, daily_timesteps=10)
#     phi_matrix16 = contact_matrix(partition16)
#     pop_s16, pop_e16, pop_i16, pop_r16 = update_start_pop(travel16)
#
#     test16 = deepcopy(params_template)
#     test16['phi'] = phi_matrix16
#     test16['start_S'] = pop_s16
#     test16['start_E'] = pop_e16
#     test16['start_I'] = pop_i16
#     test16['start_R'] = pop_r16
#     test16['n_nodes'] = 16
#
#     ref16 = deepcopy(params_template)
#     ref16['phi'] = [[[[5/10, 0], [0, 0]]]]
#     ref16_s = np.array([sorted(travel16.groupby(['age_src'])['n'].sum(), reverse=True)])
#     ref16['start_S'] = ref16_s[0, 0] - 16
#     ref16['start_E'] = np.array([[0, 0]])
#     ref16['start_I'] = np.array([[16, 0]])
#     ref16['start_R'] = np.array([[0, 0]])
#     ref16['n_nodes'] = 1
#
#     test16_model = SEIR(test16)
#     test16_model.seir()
#
#     ref16_model = SEIR(ref16)
#     ref16_model.seir()
#
#
#     # In[130]:
#
#
#     ref16_s
#
#
#     # In[131]:
#
#
#     s16_diff_max, e16_diff_max, i16_diff_max, r16_diff_max = test_partition(test16_model, ref16_model, atol=abs_tol, rtol=rel_tol)
#
#
#     # In[132]:
#
#
#     fig, ax = plt.subplots(1, 1, figsize=(8, 5))
#
#     s16_diff_max.plot(ax=ax, color='b')
#     e16_diff_max.plot(ax=ax, color='g')
#     i16_diff_max.plot(ax=ax, color='r')
#     r16_diff_max.plot(ax=ax, color='k')
#
#     plt.legend(("S", "E", "I", "R"), loc=0)
#     plt.xlabel("Time")
#     plt.xticks(rotation=45)
#     plt.title("16 Local, One Contextual vs Baseline Mixing = 5 contacts/person/day")
#     plt.tight_layout()
#     plt.ylabel("N Partition - N Baseline")
#     plt.axhline(y=0, c='gray', ls='dotted')
#     #ax.set_ylim(-2, 3)
#
#     plt.show()
