import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max

from drxnet.segments import *

# These functions are based on the implementation from roost by Rhys E. A. Goodall & Alpha A. Lee
# Source: https://github.com/CompRhys/roost

class DRXNet(nn.Module):
    """
    predict the capacity based on input information

    residue network design for cycling prediction
    """

    def __init__(
        self,
        elem_emb_len,
        elem_fea_len = 32,
        vol_fea_len = 64,
        rate_fea_len = 16,
        cycle_fea_len = 16,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        out_hidden=[256, 128, 64],
        weight_pow = 1,
        activation = nn.ReLU,
        mu = 0,
        std = 1,
        batchnorm_graph = False,
        batchnorm_condition = False,
        batchnorm_mix = False,
        batchnorm_main = False,
        **kwargs
    ):
        if isinstance(out_hidden[0], list):
            raise ValueError("boo hiss bad user")
            # assert all([isinstance(x, list) for x in out_hidden]),
            #   'all elements of out_hidden must be ints or all lists'
            # assert len(out_hidden) == len(n_targets),
            #   'out_hidden-n_targets length mismatch'

        super().__init__()

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "elem_heads": elem_heads,
            "elem_gate": elem_gate,
            "elem_msg": elem_msg,
            "cry_heads": cry_heads,
            "cry_gate": cry_gate,
            "cry_msg": cry_msg,
            "weight_pow": weight_pow,
            "activation": activation,
            "batchnorm": batchnorm_graph

        }

        self.mu = mu
        self.std = std


        self.rate_embedding = build_mlp(input_dim= 1, output_dim= rate_fea_len,
                                        hidden_dim = 2 * rate_fea_len,
                                        activation= activation, batchnorm= batchnorm_condition)

        self.cycle_embedding = build_mlp(input_dim= 1, output_dim= cycle_fea_len,
                                         hidden_dim = 2 * cycle_fea_len,
                                         activation= activation, batchnorm= batchnorm_condition)

        self.material_nn = DescriptorNetwork(**desc_dict)




        self.encode_rate = build_gate(input_dim= elem_fea_len + rate_fea_len,
                                      output_dim= elem_fea_len,
                                      activation= activation,
                                      batchnorm= batchnorm_mix)

        self.encode_cycle = build_gate(input_dim= elem_fea_len + cycle_fea_len,
                                       output_dim= elem_fea_len,
                                       activation= activation,
                                       batchnorm= batchnorm_mix)

        self.delta_N = nn.Linear(1, elem_fea_len, bias= False)

        self.encode_voltage = EncodeVoltage(hidden_dim = vol_fea_len,
                                            output_dim = elem_fea_len,
                                            activation= activation,
                                            batchnorm= batchnorm_main)

        self.add_voltage = forwardVoltage(input_dim = elem_fea_len,
                                           output_dim = elem_fea_len,
                                           activation= nn.Softplus,
                                           batchnorm= batchnorm_main)



        self.fc = build_mlp(input_dim= elem_fea_len , output_dim= 1,
                            hidden_dim = elem_fea_len, activation= nn.Softplus,
                            batchnorm= batchnorm_main)





        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        # No output neural network is required


    def forward(self,
                elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx,
                V_window, rate, cycle, Vii,
                return_direct = True,
                return_feature = False):
        """
        Forward pass through the material_nn and output_nn
        """
        # print(elem_weights)


        crys_fea, elem_fea_updated = self.material_nn(
            elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx
        )

        rate_fea = self.rate_embedding(rate)
        cycle_fea = self.cycle_embedding(cycle)

        cond_rate, attn_rate = self.encode_rate(torch.cat([crys_fea, rate_fea], dim =1))
        cond_rate = crys_fea + cond_rate

        cond_cycle, attn_cycle = self.encode_cycle(torch.cat([cond_rate, cycle_fea], dim = 1) )
        cond_cycle = cond_rate + cond_cycle * self.delta_N(cycle - 1)


        x_vol, _ = self.encode_voltage(V_window, Vii)

        ## compute the conditional feature vectors. x0 is for the first cycle
        x0 = self.add_voltage(x_vol, cond_rate)
        x = self.add_voltage(x_vol, cond_cycle)

        ## compute the capacity. q0 is for the first cycle capacity
        q0 = self.softplus(self.fc(x0))
        q_out = self.softplus(self.fc(x))


        q_grad = torch.autograd.grad(q_out, Vii, grad_outputs = torch.ones_like(q_out),
                                        create_graph=True, retain_graph=True)

        grad_weights = torch.autograd.grad(q_out, elem_weights, grad_outputs = torch.ones_like(q0),
                                        create_graph=True, retain_graph=True, allow_unused= True)



        if return_direct:
            return q_out, q_grad[0]

        elif return_feature:
            return (crys_fea, cond_rate, cond_cycle)

        else:
            return q0 , q_out, q_grad[0], torch.norm(grad_weights[0])



class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the Roost Model.
    """

    def __init__(
        self,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        weight_pow = 1,
        activation = nn.SiLU,
        batchnorm = False
    ):
        """
        """
        super().__init__()


        self.batchnorm = batchnorm
        self.activation = activation

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=elem_fea_len,
                    elem_heads=elem_heads,
                    elem_gate=elem_gate,
                    elem_msg=elem_msg,
                    weight_pow= weight_pow,
                    activation = self.activation,
                    batchnorm = self.batchnorm,
                )
                for i in range(n_graph)
            ]
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate, activation= self.activation),
                    message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg, activation= self.activation,
                                             batchnorm= self.batchnorm),
                    weight_pow = weight_pow,

                )
                for _ in range(cry_heads)
            ]
        )

    def forward(self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stoichiometry
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx

        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """

        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            )

        # head_fea = [
        #     head(elem_fea, index=cry_elem_idx, weights=elem_weights)
        #     for head in self.cry_pool
        # ]


        ## return the head-averaged pooling and the elem_fea_matrix
        return torch.mean(torch.stack(head_fea), dim=0), elem_fea

    def __repr__(self):
        return self.__class__.__name__


class MessageLayer(nn.Module):
    """
    Massage Layers are used to propagate information between nodes in
    the stoichiometry graph.
    """

    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg, weight_pow,
                 activation = nn.LeakyReLU, batchnorm = False):
        """
        """
        super().__init__()

        self.activation = activation
        self.batchnorm = batchnorm

        # Pooling and Output
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(2 * elem_fea_len, 1, elem_gate, activation = self.activation),
                    message_nn=SimpleNetwork(2 * elem_fea_len, elem_fea_len, elem_msg, activation= self.activation,
                                             batchnorm= self.batchnorm),
                    weight_pow = weight_pow,
                )
                for _ in range(elem_heads)
            ]
        )

    def forward(self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs

        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Element hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)

        # sum selectivity over the neighbours to get elems
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(
                attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights)
            )

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea

    def __repr__(self):
        return self.__class__.__name__
