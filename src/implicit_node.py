#############################################
# Calculate contact rates in implicit nodes #
#############################################

"""
Explicit nodes are nodes that have axes in the contact matrix. Implicit nodes are nodes that are implied by
contact between individuals in two different explicit nodes. For example, explicit nodes may be spatially defined,
non-overlapping areas like census block groups. The presence of schools in the model is "implied" by contact between
two or more explicit nodes that reflects some of the population in those nodes interacting.

Inputs needed
1. number of people from node A (example: 25 people)
2. number of people from node B (example: 50 people)
3. contact rate in implied node (example: 5 daily contacts per person)
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def load_travel(path):
    """
    Load the number of people traveling between nodes (both implicit and explicit)
    :return:
    """
    travel = pd.read_csv(path, header=0)
    return travel

def load_contact(path):
    """
    Load the node-independent structured per-capita daily contact matrix
    :return:
    """
    contact = pd.read_csv(path, header=0)
    return contact

def load_population(path):
    """
    Load the node-specific population sizes
    :return:
    """
    population = pd.read_csv(path, header=0)
    return population

def implied_contacts(travel_contact_df):

    travel_contact_df['expected_daily_contacts'] = travel_contact_df['n_travel'] * travel_contact_df['daily_per_capita_contacts']
    total_by_node = travel_contact_df.groupby(['destination'])['expected_daily_contacts'].sum().reset_index()

    source_nodes = travel_contact_df.groupby(['destination'])['source'].nunique().reset_index()
    source_nodes.columns = ['destination', 'number_source_nodes']

    travel_contact_df = pd.merge(travel_contact_df, total_by_node, on='destination', suffixes=['', '_destination_total'])
    travel_contact_df = pd.merge(travel_contact_df, source_nodes, on='destination')
    travel_contact_df['fraction_total_contacts'] = travel_contact_df['expected_daily_contacts'] / travel_contact_df['expected_daily_contacts_destination_total']

    # partition within node and between node contacts in the implied node
    travel_contact_df['implied_total_contacts_within'] = (travel_contact_df['fraction_total_contacts'] * travel_contact_df['expected_daily_contacts']) / travel_contact_df[
        'number_source_nodes']
    travel_contact_df['implied_total_contacts_between'] = ((1 - travel_contact_df['fraction_total_contacts']) * travel_contact_df['expected_daily_contacts']) / travel_contact_df[
        'number_source_nodes']

    # convert total contacts to per capita contacts
    travel_contact_df['implied_per_capita_contacts_within'] = travel_contact_df['implied_total_contacts_within'] / travel_contact_df['n_travel']
    travel_contact_df['implied_per_capita_contacts_between'] = travel_contact_df['implied_total_contacts_between'] / travel_contact_df['n_travel']

    # extract mappings between implicit nodes and their explicit source nodes
    implicit_dest = travel_contact_df[travel_contact_df['dest_type'] == 'implicit']

    implicit2source = {}
    for i in implicit_dest['destination'].unique():
        srcs = implicit_dest[implicit_dest['destination'] == i]
        implicit2source[i] = srcs['source'].unique()

    source2implicit = {}
    for i in implicit_dest['source'].unique():
        dests = implicit_dest[implicit_dest['source'] == i]
        source2implicit[i] = dests['destination'].unique()

    # brittle: cannot support contact stratifications other than age...
    finalized_contacts = {
        'source': [],
        'destination': [],
        'age1': [],
        'age2': [],
        'final_daily_per_capita_contact': []
    }

    # reframe the explicit -> implict contact as a combination of explicit -> explicit contacts
    for src, impl_nodes in source2implicit.items():
        # for each implicit node destination of a single explicit node
        for impl in impl_nodes:
            # get the _other_ explicit nodes that feed into the same implicit node
            other_src = [i for i in implicit2source[impl] if i != src]
            # get subset of edge list for all pairs of explict source and implied destination
            btw_contact = travel_contact_df[(travel_contact_df['source'] == src) & (travel_contact_df['destination'] == impl)]
            for i, row in btw_contact.iterrows():  # for multiple age strata in the nodes...
                # handle the within-node contacts (src->src) that happen in the implied node
                finalized_contacts['source'].append(src)
                finalized_contacts['destination'].append('{}_{}'.format(impl, src))
                finalized_contacts['age1'].append(row['age1'])
                finalized_contacts['age2'].append(row['age2'])
                # this is the crucial part: divide the between node contacts in the implict node by the number of
                # other explicit source nodes that contribute to that implicit node
                finalized_contacts['final_daily_per_capita_contact'].append(row['implied_per_capita_contacts_within'])
                for j in other_src:
                    finalized_contacts['source'].append(src)
                    finalized_contacts['destination'].append('{}_{}'.format(impl, j))
                    finalized_contacts['age1'].append(row['age1'])
                    finalized_contacts['age2'].append(row['age2'])
                    finalized_contacts['final_daily_per_capita_contact'].append(row['implied_per_capita_contacts_between'])

    return finalized_contacts

def contact_df_to_arr(cdf):

    sources = cdf['source'].unique()
    destinations = cdf['destination'].unique()

    # put named nodes in alphabetical order
    all_nodes = np.concatenate((sources, destinations)).sort()

    all_ages = ['age1', 'age2']  # 2 age groups, might as well hard-code it...

    contact_array = np.zeros([len(all_nodes), len(all_nodes), len(all_ages), len(all_ages)])

    for i, n1 in enumerate(all_nodes):
        for j, n2 in enumerate(all_nodes):
            for k, a1 in enumerate (all_ages):
                for l, a2 in enumerate(all_ages):
                    contact_val = cdf[(cdf['source'] == n1) & (cdf['destination'] == n2) & (cdf['age1'] == a1) & (cdf['age2'] == a2)]['final_daily_per_capita_contact']
                    contact_array[i, j, k, l] = contact_val

    return contact_array

def pop_df_to_arr(pop_df):

    pop_totals = pop_df.groupby(['source', 'age'])['n_travel'].sum().reset_index()
    nodes = pop_totals['source'].unique().sort()
    all_ages = ['age1', 'age2'] # again, hard-coding for simplicity's sake

    initial_pop_array = np.zeros([len(nodes)])

    for n in nodes:
        for a in all_ages:
            pass


def main():

    tr = load_travel('../inputs/travel.csv')
    co = load_contact('../inputs/contact.csv')
    pop = load_population('../inputs/all_node_population.csv')

    tc = pd.merge(tr, co, how='outer', left_on='age', right_on='age1')

    implied_travel = tc[(tc['source_type'] == 'explicit') & (tc['dest_type'] == 'implicit')]

    implied_contacts_ = implied_contacts(travel_contact_df=implied_travel)

    # now add the regular old explicit -> explicit contact to the final contacts dictionary
    expl_contact = tc[(tc['source_type'] == 'explicit') & (tc['dest_type'] == 'explicit')]
    for i, row in expl_contact.iterrows():
        implied_contacts_['source'].append(row['source'])
        implied_contacts_['destination'].append(row['destination'])
        implied_contacts_['age1'].append(row['age1'])
        implied_contacts_['age2'].append(row['age2'])
        implied_contacts_['final_daily_per_capita_contact'].append(row['daily_per_capita_contacts'])

    contact_df = pd.DataFrame.from_dict(implied_contacts_)
    #final_contact_df = contact_df.groupby(['source', 'destination', 'age1', 'age2'])['final_daily_per_capita_contact'].sum().reset_index()
    final_contact_arr = contact_df_to_arr(contact_df)
    final_pop_arr = pop_df_to_arr(tr)

    return final_contact_arr, final_pop_arr


if __name__ == '__main__':

    main()




