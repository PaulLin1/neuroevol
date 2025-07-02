"""
Functions on a genome

TO DO: Move mutate stuff here from cppn file and nn file
"""
import random

def crossover(info1, info2):
    # Equal Fitness
    # Might not implement this rn because it doesnt happen that much
    # if info1['loss'] == info2['loss']:
    #     print("Same loss")
    #     return

    # Find fitter model
    fit_model, less_fit_model = (info2['model'], info1['model']) if info1['loss'] > info2['loss'] else (info1['model'], info2['model'])

    # New model starts off as clone of more fit
    new_model = fit_model.clone()

    less_fit_conns = {conn.innov_num: conn for conn in less_fit_model.connections}

    for i in new_model.connections:
        if i.innov_num in less_fit_conns:
            # Randomly decides to inherit from less_fit if there is a matching connection
            if random.random() < 0.5:
                other_conn = less_fit_conns[i.innov_num]
                i.enable = other_conn.enable

                # Idk this is in the paper
                if not i.enable or not other_conn.enable:
                    i.enable = random.random() < 0.25  
                else:     
                    i.weight = other_conn.weight

    return new_model