from typing import Tuple
import numpy as np
from weights import Weights


class LinearLayer:
    def __init__(self, size, activation=None, prev_layer=None, next_layer=None):
        self._activation: Tuple[function, function] = activation or (None, None)
        self._delta_coef: np.ndarray = None
        self._mask: np.ndarray = None
        self._prev_layer: LinearLayer = None
        self._size: int = size
        self._values: np.ndarray = None
        self._weights: Weights = None

        if (prev_layer is not None):
            self.prev_layer = prev_layer

        if (next_layer is not None):
            next_layer.prev_layer = self
    
    @property
    def activation(self):
        return self._activation[0] or (lambda x: x)

    @property
    def activation_deriv(self):
        return self._activation[1] or (lambda x: 1)

    @property
    def delta_coef(self):
        return self._delta_coef
        
    @delta_coef.setter
    def delta_coef(self, delta):
        self._delta_coef = delta.reshape(-1, *self.output_shape)

    @property
    def input_shape(self):
        return self.prev_layer.output_shape

    @property
    def linear_size(self):
        return np.prod(self.output_shape)

    @property
    def mask(self):
        return self._mask
        
    @mask.setter
    def mask(self, new_mask):
        self._mask = new_mask
        
    @property
    def output_shape(self):
        return self._size,

    @property
    def prev_layer(self):
        return self._prev_layer
    
    @prev_layer.setter
    def prev_layer(self, prev_layer):
        self._prev_layer = prev_layer

        self.weights = Weights(
            shape=(self.prev_layer.linear_size, self.linear_size),
            scale_coef=(1/self.prev_layer.linear_size)
        )

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        self._values = self.activation(new_values)

    @property
    def values_with_mask(self):
        self.mask = self._create_mask()
        return self.values * self.mask

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights: Weights):
        self._weights = new_weights

    def back_propagation(self):
        assert self.prev_layer is not None

        self.prev_layer.delta_coef = (
            np.dot(self.delta_coef, self.weights.values.T)
            * self.prev_layer.activation_deriv(self.prev_layer.values)
            * self.prev_layer.mask
        )

    def update_values(self, mask=False):
        self.prev_layer.values = self.prev_layer.values.reshape(self.prev_layer.values.shape[0], -1)
        if (mask):
            self.values = (
                np.dot(self.prev_layer.values_with_mask, self.weights.values)
            )
        else:
            self.values = np.dot(self.prev_layer.values, self.weights.values)

    def update_weights(self, alpha):
        weights_delta = np.dot(
            self.prev_layer.values.T, self.delta_coef
        ) * alpha

        self.weights.values -= weights_delta

    def _create_mask(self):
        return np.random.randint(2, size=self.values.shape) * 2
