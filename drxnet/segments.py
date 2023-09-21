import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean

# These functions are based on the implementation from roost by Rhys E. A. Goodall & Alpha A. Lee
# Source: https://github.com/CompRhys/roost


class MeanPooling(nn.Module):
    """Mean pooling"""

    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return scatter_mean(x, index, dim=0)

    def __repr__(self):
        return self.__class__.__name__


class SumPooling(nn.Module):
    """Sum pooling"""

    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return scatter_add(x, index, dim=0)

    def __repr__(self):
        return self.__class__.__name__


class AttentionPooling(nn.Module):
    """
    softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Args:
            gate_nn: Variable(nn.Module)
            message_nn
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, x, index):
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__


class WeightedAttentionPooling(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn, weight_pow = 1):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

        self.pow = weight_pow # torch.nn.Parameter(torch.randn(1))

    def forward(self, x, index, weights):
        gate = self.gate_nn(x)

        # S_max = scatter_max(gate, index, dim=0)[0]

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = (weights ** self.pow) * gate.exp()
        # gate = weights * gate.exp()
        # gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dims,
        activation=nn.LeakyReLU,
        batchnorm=False,
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()

        self.fc_out.reset_parameters()


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dims,
        activation=nn.ReLU,
        batchnorm=False,
        return_features=False,
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.return_features = return_features
        if not self.return_features:
            self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        if self.return_features:
            return x
        else:
            return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__



class EncodeVoltage(nn.Module):
    """
    Encode voltage window with [V_low, V_high] and output a mixed feature with Vii as input
    """

    def __init__(
        self,
        hidden_dim = 64,
        output_dim = 32,
        activation=nn.Softplus,
        batchnorm=False,
    ):
        """
        Inputs
        ----------
        output_dim: int

        """
        super().__init__()


        self.fc_window = nn.Linear(2, hidden_dim)
        self.fc_Vii = nn.Linear(1, hidden_dim)

        if batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = nn.Identity()

        self.act = activation()
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, V_window, Vii):
        """
        :param V_window: tensor of size [None, 2]
        :param Vii: tensor of size [None, 1]
        :return:
        """

        window_fea = self.fc_window(V_window)
        out = self.fc(self.act(self.bn( window_fea  + self.fc_Vii(Vii) )))
        return out, window_fea


    def __repr__(self):
        return self.__class__.__name__


class EncodeDiff(nn.Module):
    """
    Encode voltage window with [V_low, V_high] and output a mixed feature with Vii as input
    """

    def __init__(
        self,
        output_dim = 64,
        activation=nn.Softplus,
        batchnorm=False,
    ):
        """
        Inputs
        ----------
        output_dim: int

        """
        super().__init__()


        self.fc = nn.Linear(1, output_dim)

        if batchnorm:
            self.bn = nn.BatchNorm1d(output_dim)
        else:
            self.bn = nn.Identity()

        self.act = activation()


    def forward(self, x_diff, f_norm):
        """
        :param V_window: tensor of size [None, 2]
        :param Vii: tensor of size [None, 1]
        :return:
        """
        out = self.act(self.bn( x_diff + self.fc(f_norm) ))
        return out


    def __repr__(self):
        return self.__class__.__name__

class build_gate(nn.Module):
    """
    Use simple network to forward information
    """

    def __init__(
        self,
        input_dim = 64,
        output_dim = 32,
        activation=nn.Softplus,
        batchnorm=False,
    ):
        """
        Inputs
        ----------
        output_dim: int

        """
        super().__init__()


        self.kernel = nn.Linear(input_dim, output_dim)
        self.attn = nn.Linear(input_dim, 1)

        if batchnorm:
            self.bn = nn.BatchNorm1d(output_dim)
            self.bn_attn = nn.BatchNorm1d(1)
        else:
            self.bn = nn.Identity()
            self.bn_attn = nn.Identity()

        self.act = activation()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        """
        attn_coef = self.sigmoid(self.bn_attn(self.attn(x)))
        out = attn_coef * self.act(self.bn( self.kernel(x) ))
        return out, attn_coef


    def __repr__(self):
        return self.__class__.__name__


class build_mlp(nn.Module):
    """
    Use simple network to forward information
    """

    def __init__(
        self,
        input_dim = 64,
        hidden_dim = 32,
        output_dim = 32,
        activation=nn.Softplus,
        batchnorm=False,
    ):
        """
        Inputs
        ----------
        output_dim: int

        """
        super().__init__()


        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = nn.Identity()

        self.act = activation()


    def forward(self, x):
        """
        """
        out = self.fc(self.act(self.bn( self.hidden(x) )))
        return out


    def __repr__(self):
        return self.__class__.__name__


class forwardVoltage(nn.Module):
    """
    Forward voltage information with feature vector addition
    """

    def __init__(
        self,
        input_dim = 32,
        output_dim = 32,
        activation=nn.Softplus,
        batchnorm=False,
    ):
        """
        Inputs
        ----------
        output_dim: int

        """
        super().__init__()

        self.hidden = nn.Linear(input_dim, output_dim)

        if batchnorm:
            self.bn = nn.BatchNorm1d(output_dim)
        else:
            self.bn = nn.Identity()

        self.act = activation()


    def forward(self, x, add_layer):
        """
        """
        out = self.act(self.bn( self.hidden( x + add_layer )))
        return out


    def __repr__(self):
        return self.__class__.__name__
