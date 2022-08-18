from linearlayer import LinearLayer
from typing import Tuple
import numpy as np
from weights import Weights


class ConvLayer(LinearLayer):
    def __init__(self, kernel_size, kernel_output_size, padding=0, stride=1, activation=None, prev_layer=None, next_layer=None):
        self._activation: Tuple[function, function] = activation or (None, None)
        self._delta_coef: np.ndarray = None
        self._kernel_size: int = kernel_size
        self._kernel_output_size: int = kernel_output_size
        self._mask: np.ndarray = None
        self._padding: int = padding
        self._prev_layer: LinearLayer = None
        self._prev_layer_areas: np.ndarray = None
        self._stride: int = stride
        self._values: np.ndarray = None
        self._weights: Weights = None

        assert prev_layer is not None
        assert next_layer is not None
        self.prev_layer = prev_layer
        next_layer.prev_layer = self

        # assert self.output_image_size == self.image_size

    @property
    def image_size(self):
        assert self.input_shape[1] == self.input_shape[2]
        return self.input_shape[2]

    @property
    def image_channels(self):
        return self.input_shape[0]

    @property
    def kernel_output_size(self):
        return self._kernel_output_size

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def output_image_size(self):
        return (self.image_size + 2*self.padding - self.kernel_size) // self.stride + 1
        # self.image_size = (self.output_image_size - 1) * self.stride - 2*self.padding + self.kernel_size

    @property
    def output_shape(self):
        return self.kernel_output_size * self.image_channels, self.output_image_size, self.output_image_size

    @property
    def padding(self):
        return self._padding

    @property
    def prev_layer(self):
        return super().prev_layer

    @prev_layer.setter
    def prev_layer(self, prev_layer):
        self._prev_layer = prev_layer

        self.weights = Weights(
            shape=(self.kernel_size**2, self.kernel_output_size),
            scale_coef=1/self.kernel_size**2
        )

    @property
    def stride(self):
        return self._stride

    def back_propagation(self):
        d = self.delta_coef

        d = d.reshape(
            -1,
            self.image_channels,
            self.kernel_output_size,
            self.output_image_size,
            self.output_image_size
        )

        d = d.swapaxes(2, 3)
        d = d.swapaxes(3, 4)
        d = d.reshape(np.prod(d.shape[:-1]), d.shape[4])
        d = np.dot(d, self.weights.values.T)

        # d_mxs = np.argmax(np.abs(d), axis=1)[:, np.newaxis]

        # d = np.take_along_axis(d, d_mxs, axis=1)
 
        # d = d.reshape(
        #     -1,
        #     self.image_channels,
        #     self.output_image_size,
        #     self.output_image_size
        # )

        d = d.reshape(
            -1,
            self.image_channels,
            self.output_image_size,
            self.output_image_size,
            self.kernel_size,
            self.kernel_size
        )

        new_d = np.zeros(
            shape=(d.shape[0], self.image_channels, self.image_size + 2 * self.padding, self.image_size + 2 * self.padding)
        )

        for i in range(d.shape[0]):
            for j in range(self.image_channels):
                for k in range(self.output_image_size):
                    for l in range(self.output_image_size):
                        for m in range(self.kernel_size):
                            for n in range(self.kernel_size):
                                a = new_d[i, j, k * self.stride + m, l * self.stride + n]
                                b = d[i, j, k, l, m, n]
                                new_d[i, j, k * self.stride + m, l * self.stride + n] = a if abs(a) > abs(b) else b
        
        d = new_d[:, :, self.padding: new_d.shape[2]-self.padding, self.padding: new_d.shape[3]-self.padding]

        # print(
        #     f"{type(self)} : delta shape for a previous layer is {d.shape}"
        # )

        self.prev_layer.delta_coef = (
            d
            * self.prev_layer.activation_deriv(self.prev_layer.values) 
            * self.prev_layer.mask
        )

    def get_areas(self, inputs):
        assert len(inputs.shape) == 4  # count x channels x height x width, width == height == size

        count = inputs.shape[0]
        channels = inputs.shape[1]

        assert channels == self.image_channels
        assert inputs.shape[2] == inputs.shape[3] == self.image_size

        for _ in range(self.padding):
            inputs = np.insert(inputs, 0, inputs[:, :, 0, :], axis=2)
            inputs = np.insert(inputs, -1, inputs[:, :, -1, :], axis=2)

            inputs = np.insert(inputs, 0, inputs[:, :, :, 0], axis=3)
            inputs = np.insert(inputs, -1, inputs[:, :, :, -1], axis=3)

        result = []
        for row_start in range(0, self.image_size - self.kernel_size + 2 * self.padding + 1, self.stride):
            for col_start in range(0, self.image_size - self.kernel_size + 2 * self.padding + 1, self.stride):
                result.append(
                    inputs[
                        :, :,
                        row_start: row_start + self.kernel_size,
                        col_start: col_start + self.kernel_size,
                    ].reshape(count, channels, self.kernel_size**2)
                )

        result = np.array(result).swapaxes(0, 1)
        result = result.swapaxes(1, 2)

        assert result.shape == (count, self.image_channels, self.output_image_size**2, self.kernel_size**2)

        return result

    def update_values(self, mask=False):
        if (mask):
            inputs = self.prev_layer.values_with_mask
        else:
            inputs = self.prev_layer.values

        inputs = self.get_areas(inputs)
        self._prev_layer_areas = inputs

        assert inputs.shape[1] == self.image_channels
        assert inputs.shape[2] == self.output_image_size**2
        assert inputs.shape[3] == self.kernel_size**2

        inputs = inputs.reshape(np.prod(inputs.shape[:-1]), self.kernel_size**2)
        self.values = np.dot(inputs, self.weights.values)
        
        self.values = self.values.reshape(
            -1,
            self.image_channels,
            self.output_image_size**2,
            self.kernel_output_size
        )

        self.values = self.values.swapaxes(2, 3)
        
        self.values = self.values.reshape(
            -1,
            self.image_channels * self.kernel_output_size,
            self.output_image_size,
            self.output_image_size
        )

    def update_weights(self, alpha):
        d = self.delta_coef

        d = d.reshape(
            -1,
            self.image_channels,
            self.kernel_output_size,
            self.output_image_size,
            self.output_image_size
        )

        d = d.swapaxes(2, 3)
        d = d.swapaxes(3, 4)
        d = d.reshape(np.prod(d.shape[:-1]), d.shape[4])

        v = self._prev_layer_areas
        v = v.reshape(v.shape[0] * v.shape[1] * v.shape[2], v.shape[3])

        weights_delta = np.dot(v.T, d) * alpha

        self.weights.values -= weights_delta

