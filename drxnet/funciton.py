from torch.utils.data import DataLoader
from drxnet.core import Featurizer
from drxnet.drxnet.model import DRXNet

import os
import torch
import torch.nn as nn
import numpy as np


from pymatgen.core import Composition


class voltageFeaturizer():
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(
        self,
        fea_path = None,
        ):

        if fea_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.elem_features = Featurizer.from_json(os.path.join(current_dir, "./data/el-embeddings/matscholar-embedding.json"))
        else:
            self.elem_features = Featurizer.from_json(fea_path)

        self.elem_emb_len = self.elem_features.embedding_size


    def get_inputs(self,
            input_composition = 'Li1.2Mn0.4Ti0.4O2.0',
            input_V_low = 1.5,
            input_V_high = 4.8,
            input_rate = 10,
            input_cycle = 1,
            input_Vii = 2.11,
            input_cry_ids = 0):

        composition = input_composition

        V_low = input_V_low
        V_high =  input_V_high
        rate = input_rate
        cycle = input_cycle
        Vii = input_Vii

        cry_ids = input_cry_ids

        comp_dict = Composition(composition).get_el_amt_dict()
        F_content = (2.0 - comp_dict['O']) / 2

        try:
            comp_dict.pop('F')
            comp_dict.pop('O')
        except:
            comp_dict.pop('O')

#         print(comp_dict)
        elements = list(comp_dict.keys())

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / 2.0

        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) + self.elem_features.get_fea('F') * F_content for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_ids[0]} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_ids[0]} [{composition}] composition cannot be parsed into elements"
            )

        nele = len(elements)
        self_fea_idx = []
        nbr_fea_idx = []
        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nele
            nbr_fea_idx += list(range(nele))

        # convert all data to tensors
        atom_weights = torch.tensor(weights, requires_grad = True, dtype= torch.float32)# torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        V_window = torch.Tensor([V_low, V_high])

        rate = torch.Tensor([rate])
        cycle = torch.Tensor([cycle])

#         print(rate, cycle)

#
        Vii = torch.tensor([Vii], requires_grad = True, dtype= torch.float32)

        return ((atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, V_window,
                rate, cycle, Vii), cry_ids )


def collate_profile_v1(
                        featurizer,
                        num_points = 100,
                        input_composition = 'Li1.2Mn0.4Ti0.4O2.0',
                        input_V_low = 1.5,
                        input_V_high = 4.8,
                        input_rate = 10,
                        input_cycle = 1,
                        input_Vii = 2.11,
                        input_cry_ids = 0):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    batch_window = []
    batch_rate = []
    batch_cycle = []
    batch_Vii = []
    crystal_atom_idx = []
    batch_targets = []
    batch_cry_ids = []

    cry_base_idx = 0

    V_series = np.linspace(input_V_low, input_V_high, num_points)

    for ii in range(num_points):
        inputs_, cry_id = featurizer.get_inputs(input_composition = input_composition,
                                        input_V_low = input_V_low,
                                        input_V_high = input_V_high,
                                        input_rate = input_rate,
                                        input_cycle = input_cycle,
                                        input_Vii = V_series[ii],
                                        input_cry_ids = ii)

        atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, V_window, rate, cycle, Vii = inputs_

        # number of atoms for this crystal
        n_i = atom_fea.shape[0]



        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)
        batch_window.append(V_window)
        batch_rate.append(rate)
        batch_cycle.append(cycle)
        batch_Vii.append(Vii)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([ii] * n_i))

        # increment the id counter
        cry_base_idx += n_i


    return (
        (
            torch.cat(batch_atom_weights, dim=0),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx),
            torch.stack(batch_window),
            torch.stack(batch_rate),
            torch.stack(batch_cycle),
            torch.stack(batch_Vii)
        ),
        *zip(*batch_cry_ids),
    )


def load_pretrained_models():
    model_list = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "./pretrained/")
    file_list = os.listdir(model_dir)

    for model_file in file_list:
        if not ("model" in model_file):
            continue

        model_name = os.path.join(model_dir + model_file)
        model = DRXNet(elem_emb_len= 200, elem_fea_len = 32, vol_fea_len = 64,
                               rate_fea_len = 16, cycle_fea_len = 16,
                               n_graph = 3,
            elem_heads=3,
            elem_gate=[64],
            elem_msg=[64],
            cry_heads=3,
            cry_gate=[64],
            cry_msg=[64],
            activation = nn.SiLU,
            batchnorm_graph = False,
            batchnorm_condition = True,
            batchnorm_mix = True,
            batchnorm_main = False
            )
        model.eval()
        model.load_state_dict(torch.load(model_name , map_location=torch.device('cpu')))

        model_list.append(model)

    return model_list


def ensemble_prediction(model_list, vol_featurizer, composition_string, input_V_low, input_V_high,
                        input_rate, input_cycle):
    Q_ensemble = np.empty((100,1))

    for model in model_list:
        (inputs, *_) = collate_profile_v1(vol_featurizer,
                                  input_composition= composition_string,
                                  input_rate = input_rate,
                                  input_cycle= input_cycle,
                                  input_V_low = input_V_low,
                                  input_V_high = input_V_high
                                )

        Q_, dQdV_, *_ = model(*inputs, return_direct = True)
        V_ = inputs[8]

        Q_ensemble = np.hstack([Q_ensemble, Q_.detach().numpy()])


    Q_avg = np.average(Q_ensemble[:,1:], axis = 1)
    V_avg = V_.detach().numpy()[:, 0]
    Q_std = np.std(Q_ensemble[:,1:], axis = 1)

    return Q_avg, V_avg, Q_std


def predict_V_E(Q_, V_):

    Q_max = Q_[0]
    dQ = np.gradient(Q_)
    V = V_
    dE = np.flip(dQ* V)


    E = np.zeros(len(dE))
    for i in range(len(dE)):
        E[i] = np.sum(-dE[:i])

    E = np.flip(E)

    E_avg = E[0]
    V_avg = E_avg / Q_max

    return V_avg, E_avg


def validate_V_E(Q_, V_):
    Q_max = Q_[0]
    dQ = np.gradient(Q_)
    V = V_
    dE = np.flip(dQ* V)


    E = np.zeros(len(dE))
    for i in range(len(dE)):
        E[i] = np.sum(-dE[:i])

    E = np.flip(E)

    E_avg = E[0]
    V_avg = E_avg / Q_max

    return V_avg, E_avg
