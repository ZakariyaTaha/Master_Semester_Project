'''
from . import isbi12
from . import tabea_neurons
from . import neurons
from . import neurons_noise1
from . import neurons_noise2
from . import neurons_noise3
from . import neurons_crops
from . import synthetic
# from . import roads
# from . import deepglobe
# from . import spacenet
# from . import canal
from . import carls
from . import mra
'''
from . import myDataset
import numpy as np

def load_dataset(dataset_name, sequence='training', size="orig", labels="all", each=1, graph=False, threshold=15, brains=None, clip_value=None):
    '''
    if dataset_name == 'isbi12':
        data_points = isbi12.load_dataset(sequence, size, labels, each, graph, threshold)
    elif dataset_name == 'tabea':
        data_points = tabea_neurons.load_dataset(sequence, size, labels, each, graph, threshold)
    elif dataset_name == 'neuron':
        data_points = neurons.load_dataset(sequence, size, labels, each, graph, threshold)
    elif dataset_name == 'neuron_noise1':
        data_points = neurons_noise1.load_dataset(sequence, size, labels, each, graph, threshold)
    elif dataset_name == 'neuron_noise2':
        data_points = neurons_noise2.load_dataset(sequence, size, labels, each, graph, threshold)
    elif dataset_name == 'neuron_noise3':
        data_points = neurons_noise3.load_dataset(sequence, size, labels, each, graph, threshold)
    elif dataset_name == 'neuron_crops':
        data_points = neurons_crops.load_dataset(sequence, size, labels, each, graph, threshold)
    elif dataset_name == 'carls':
        data_points = carls.load_dataset(sequence, size, labels, each, graph, threshold)
    elif dataset_name == 'mra':
        data_points = mra.load_dataset(sequence, size, labels, each, graph, threshold)
    elif dataset_name == 'synth':
        data_points = synthetic.load_dataset(sequence, size, labels, each, graph, threshold)
    '''
    if dataset_name == 'myDataset':
        data_points = myDataset.load_dataset(brains, clip_value, sequence, size, labels, each, threshold)
    else:
        raise ValueError("unknown dataset '{}'".format(dataset_name))

    return data_points
