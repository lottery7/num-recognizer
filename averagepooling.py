from convlayer import ConvLayer
import numpy as np


class AveragePooling:
    def __init__(self, filter_size, stride, prev_layer, next_layer):
        self._filter_size: int = filter_size
        self._prev_layer: ConvLayer = None
        self._stride: int = stride
        self._values: np.ndarray = None

        self.prev_layer = prev_layer
        next_layer.prev_layer = self
    
    @property
    def activation_deriv(self):
        return lambda x: 1

    @property
    def delta_coef(self):
        return self._delta_coef
    
    @delta_coef.setter
    def delta_coef(self, new_delta_coef):
        self._delta_coef = new_delta_coef.reshape(-1, *self.output_shape)

    @property
    def filter_size(self):
        return self._filter_size
    
    @property
    def image_channels(self):
        return self.prev_layer.output_shape[0]
    
    @property
    def image_size(self):
        return self.prev_layer.output_image_size
    
    @property
    def input_shape(self):
        return self.prev_layer.output_shape
        
    @property
    def linear_size(self):
        return np.prod(self.output_shape)
    
    @property
    def mask(self):
        return 1

    @property
    def output_image_size(self):
        return (self.image_size - self.filter_size) // self.stride + 1

    @property
    def output_shape(self):
        return self.prev_layer.output_shape[0], self.output_image_size, self.output_image_size
    
    @property
    def prev_layer(self):
        return self._prev_layer
    
    @prev_layer.setter
    def prev_layer(self, new_layer):
        assert isinstance(new_layer, ConvLayer), "Pooling layer can be applied to a convolutional layer only"
        self._prev_layer = new_layer
    
    @property
    def stride(self):
        return self._stride

    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, new_values):
        self._values = new_values
    
    @property
    def values_with_mask(self):
        return self.values

    def back_propagation(self):
        d = self.delta_coef

        tmp = np.zeros(shape=(d.shape[0], *self.prev_layer.output_shape))

        for row_start in range(0, self.image_size - self.filter_size + 1, self.stride):
            for col_start in range(0, self.image_size - self.filter_size + 1, self.stride):
                for i in range(self.filter_size):
                    for j in range(self.filter_size):
                        tmp[:, :, row_start + i, col_start + j] = d[:, :, row_start // self.stride, col_start // self.stride]
                
        d = tmp.reshape(-1, *self.prev_layer.output_shape)

        self.prev_layer.delta_coef = (
            d
            * self.prev_layer.activation_deriv(self.prev_layer.values) 
            * self.prev_layer.mask
        )
    
    def get_areas(self, inputs):
        assert len(inputs.shape) == 4  # count x channels x height x width, width == height == size

        count = inputs.shape[0]
        channels = inputs.shape[1]

        assert inputs.shape[1:] == self.prev_layer.output_shape

        result = []
        for row_start in range(0, self.image_size - self.filter_size + 1, self.stride):
            for col_start in range(0, self.image_size - self.filter_size + 1, self.stride):
                result.append(
                    inputs[
                        :, :,
                        row_start: row_start + self.filter_size,
                        col_start: col_start + self.filter_size,
                    ].reshape(count, channels, self.filter_size**2)
                )

        result = np.swapaxes(np.array(result), 0, 1)
        result = np.swapaxes(result, 1, 2)

        assert result.shape == (count, channels, self.output_image_size**2, self.filter_size**2)

        return result
    
    def update_values(self, mask=False):
        if (mask):
            inputs = self.prev_layer.values_with_mask
        else:
            inputs = self.prev_layer.values

        inputs = self.get_areas(inputs)
        count = inputs.shape[0]
        channels = inputs.shape[1]
        inputs = np.mean(inputs, axis=3, keepdims=True)
        
        self.values = inputs.reshape(
            count, channels, self.output_image_size, self.output_image_size
        )

    def update_weights(self, *args, **kwargs):
        return
