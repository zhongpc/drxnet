import functools
import os

import numpy as np
import pandas as pd
import torch
from pymatgen.core.composition import Composition
from torch.utils.data import Dataset

from drxnet.core import Featurizer

class BatteryData_(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(
        self,
        data_path,
        fea_path,
        identifiers=["material_id", "composition"],
        add_noise = False,
        # identifiers=["material_id", "composition"],
    ):
        """[summary]

        Args:
            data_path (str): [description]
            fea_path (str): [description]
            task_dict ({name: task}): list of tasks
            identifiers (list, optional): column names for unique identifier
                and pretty name. Defaults to ["id", "composition"].
        """

        assert len(identifiers) == 2, "Two identifiers are required"

        self.identifiers = identifiers
        self.add_noise = add_noise

        assert os.path.exists(data_path), f"{data_path} does not exist!"
        # NOTE make sure to use dense datasets,
        # NOTE do not use default_na as "NaN" is a valid material

        data_list = []
        for filename in os.listdir(data_path):
            if '.csv' in filename:
                print(filename)
                df = pd.read_csv(os.path.join(data_path, filename), keep_default_na=False, na_values=[])
                data_list.append(df)

        self.df = pd.concat(data_list, axis=0, ignore_index=True)



        assert os.path.exists(fea_path), f"{fea_path} does not exist!"
        self.elem_features = Featurizer.from_json(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size



    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache data for faster training
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx (int): dataset index

        Raises:
            AssertionError: [description]
            ValueError: [description]

        Returns:
            atom_weights: torch.Tensor shape (M, 1)
                weights of atoms in the material
            atom_fea: torch.Tensor shape (M, n_fea)
                features of atoms in the material
            self_fea_idx: torch.Tensor shape (M*M, 1)
                list of self indices
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbor indices
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material

        """
        df_idx = self.df.iloc[idx]
        composition = df_idx[['composition']][0]
        V_low = df_idx[['V_low']][0]
        V_high =  df_idx[['V_high']][0]
        rate = df_idx[['rate']][0]
        cycle = df_idx[['cycle']][0]
        Vii = df_idx[['Vii']][0]

        cry_ids = df_idx[self.identifiers].values
        comp_dict = Composition(composition).get_el_amt_dict()


        if self.add_noise:
            F_content = (2.0 - comp_dict['O']) / 2 + np.random.rand(1) * 1e-3
            F_content = np.clip(F_content, 0, 1)
        else:
            F_content = (2.0 - comp_dict['O']) / 2

        try:
            comp_dict.pop('F')
            comp_dict.pop('O')
        except:
            comp_dict.pop('O')

        elements = list(comp_dict.keys())

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / 2

        if self.add_noise:
            weights += np.random.rand(len(weights), 1) * 1e-3
            weights = np.clip(weights, 0, 1)

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
        atom_weights = torch.tensor(weights, requires_grad = True, dtype= torch.float32) # torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        V_window = torch.Tensor([V_low, V_high])
        rate = torch.Tensor([rate])
        cycle = torch.Tensor([cycle])
        Vii = torch.tensor([Vii], requires_grad = True, dtype= torch.float32)

        Q0ii = df_idx['Q0ii']
        Qii = df_idx['Qii']
        dQdVii = df_idx['dQdVii']

        targets = [torch.Tensor([Q0ii]), torch.Tensor([Qii]), torch.Tensor([dQdVii]) ]
        # for target in self.task_dict:
        #     if self.task_dict[target] == "regression":
        #         targets.append(torch.Tensor([df_idx[target]]))
        #     elif self.task_dict[target] == "classification":
        #         targets.append(torch.LongTensor([df_idx[target]]))

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, V_window, rate, cycle,  Vii),
            targets,
            *cry_ids,
        )



class read_battery_infor(Dataset):
    """
    Return the battery test information
    """

    def __init__(
        self,
        data_path,
        fea_path,
        identifiers=["material_id", "composition"],
        add_noise = False,
        # identifiers=["material_id", "composition"],
    ):
        """[summary]

        Args:
            data_path (str): [description]
            fea_path (str): [description]
            task_dict ({name: task}): list of tasks
            identifiers (list, optional): column names for unique identifier
                and pretty name. Defaults to ["id", "composition"].
        """

        assert len(identifiers) == 2, "Two identifiers are required"

        self.identifiers = identifiers
        self.add_noise = add_noise

        assert os.path.exists(data_path), f"{data_path} does not exist!"
        # NOTE make sure to use dense datasets,
        # NOTE do not use default_na as "NaN" is a valid material

        data_list = []
        for filename in os.listdir(data_path):
            if '.csv' in filename:
                print(filename)
                df = pd.read_csv(os.path.join(data_path, filename), keep_default_na=False, na_values=[])
                first_row = df.iloc[1]

                print(first_row)
                data_list.append(first_row)

        self.df = pd.concat(data_list, axis=0, ignore_index=True)



        assert os.path.exists(fea_path), f"{fea_path} does not exist!"
        self.elem_features = Featurizer.from_json(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size



    def __len__(self):
        return len(self.df)

    def get_df(self):
        return self.df

    @functools.lru_cache(maxsize=None)  # Cache data for faster training
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx (int): dataset index

        Raises:
            AssertionError: [description]
            ValueError: [description]

        Returns:
            atom_weights: torch.Tensor shape (M, 1)
                weights of atoms in the material
            atom_fea: torch.Tensor shape (M, n_fea)
                features of atoms in the material
            self_fea_idx: torch.Tensor shape (M*M, 1)
                list of self indices
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbor indices
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material

        """
        df_idx = self.df.iloc[idx]
        composition = df_idx[['composition']][0]
        V_low = df_idx[['V_low']][0]
        V_high =  df_idx[['V_high']][0]
        rate = df_idx[['rate']][0]
        cycle = df_idx[['cycle']][0]
        Vii = df_idx[['Vii']][0]

        cry_ids = df_idx[self.identifiers].values
        comp_dict = Composition(composition).get_el_amt_dict()


        if self.add_noise:
            F_content = (2.0 - comp_dict['O']) / 2 + np.random.rand(1) * 1e-3
            F_content = np.clip(F_content, 0, 1)
        else:
            F_content = (2.0 - comp_dict['O']) / 2

        try:
            comp_dict.pop('F')
            comp_dict.pop('O')
        except:
            comp_dict.pop('O')

        elements = list(comp_dict.keys())

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / 2

        if self.add_noise:
            weights += np.random.rand(len(weights), 1) * 1e-3
            weights = np.clip(weights, 0, 1)

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
        atom_weights = torch.tensor(weights, requires_grad = True, dtype= torch.float32) # torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        V_window = torch.Tensor([V_low, V_high])
        rate = torch.Tensor([rate])
        cycle = torch.Tensor([cycle])
        Vii = torch.tensor([Vii], requires_grad = True, dtype= torch.float32)

        Q0ii = df_idx['Q0ii']
        Qii = df_idx['Qii']
        dQdVii = df_idx['dQdVii']

        targets = [torch.Tensor([Q0ii]), torch.Tensor([Qii]), torch.Tensor([dQdVii]) ]
        # for target in self.task_dict:
        #     if self.task_dict[target] == "regression":
        #         targets.append(torch.Tensor([df_idx[target]]))
        #     elif self.task_dict[target] == "classification":
        #         targets.append(torch.LongTensor([df_idx[target]]))

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, V_window, rate, cycle,  Vii),
            targets,
            *cry_ids,
        )




def collate_batch_(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      self_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_weights: torch.Tensor shape (N, 1)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_self_fea_idx: torch.LongTensor shape (N, M)
        Indices of mapping atom to copies of itself
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_comps: list
    batch_ids: list
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
    for i, (inputs, target, *cry_ids) in enumerate(dataset_list):
        atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, V_window, rate, cycle, Vii = inputs

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
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)

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
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        *zip(*batch_cry_ids),
    )
