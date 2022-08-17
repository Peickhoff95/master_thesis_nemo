import math
from collections import OrderedDict

import ipdb
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType


__all__ = ['HighwayLayerNetwork']

class HighwayNetwork(NeuralModule):

    """
    HighwayNetwork for Spectogram Reconstruction

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers
        normal_bias (bool): enables bias for normal linear layers
            Defaults to True
        normal_activation (torch.nn.functional): activation function for normal layer
            Defaults to torch.nn.relu
        gate_activation (torch.nn.functional): activation function for gate layer
            Defaults to torch.nn.sigmoid
        gate_bias (float): set initial constant for gate linear layer bias (should be negative)
            Defaults to -1.
    """

    def input_example(self, max_batch=None, max_dim=None):
        return None

    @property
    def input_types(self):
        return None

    @property
    def output_types(self):
        return None

    def __init__(
            self,
            feat_in,
            n_layers,
            normal_bias=True,
            normal_activation=F.relu,
            gate_activation=torch.sigmoid,
            gate_bias=-1.
    ):
        super().__init__()

        self._feat_in = feat_in

        self.normal_activation = normal_activation
        self.gate_activation = gate_activation

        self.normal_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        for i in range(n_layers):
            normal_layer = nn.Linear(self._feat_in, self._feat_in, bias=normal_bias)
            self.normal_layers.append(normal_layer)
            gate_layer = nn.Linear(self._feat_in, self._feat_in)
            gate_layer.bias.data.fill_(gate_bias)
            self.gate_layers.append(gate_layer)

    #TODO: Look up typecheck
    @typecheck()
    def forward(self, encoder_output):
        x = encoder_output
        for normal_layer, gate_layer in zip(self.normal_layers, self.gate_layers):
            normal_layer_output = self.normal_activation(normal_layer(x))
            gate_layer_output = self.gate_activation(gate_layer(x))

            transform_output = torch.mul(normal_layer_output, gate_layer_output)
            carry_output = torch.mul(x, (1-gate_layer_output))
            x = torch.add(transform_output, carry_output)

        return x



class ReconstructionDecoder(NeuralModule, Exportable):
    """
    Reconstruction Decoder
    """

    @property
    def input_types(self):
        return None

    @property
    def output_types(self):
        return None

    def __init__(self,
                 feat_in,
                 feat_out,
                 n_networks,
                 n_layers,
                 normal_bias=True,
                 normal_activation=F.relu,
                 gate_activation=torch.sigmoid,
                 gate_bias=-1.
    ):

        super().__init__()

        self._feat_in = feat_in
        self._feat_out = feat_out

        self.projection_layers = torch.nn.ModuleList()

        self.highway_networks = torch.nn.ModuleList()
        for i in range(n_networks):
            network = HighwayNetwork(
                feat_in=feat_out,
                n_layers=n_layers,
                normal_bias=normal_bias,
                normal_activation=normal_activation,
                gate_activation=gate_activation,
                gate_bias=gate_bias,
            )
            self.highway_networks.append(network)
            self.projection_layers.append(nn.Linear(feat_in, feat_out))

    @typecheck()
    def forward(self, encoder_output):
        encoder_output = torch.transpose(encoder_output, 1, 2)
        encoder_output = self.projection(encoder_output)
        network_outputs = []
        for network, projection in zip(self.highway_networks, self.projection_layers):
            network_outputs.append(network(projection(encoder_output)))
        decoder_output = torch.cat(network_outputs, -1)
        decoder_output = decoder_output.reshape([decoder_output.shape[0],-1,self._feat_out])
        return torch.transpose(decoder_output, 1, 2)

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """

        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])
