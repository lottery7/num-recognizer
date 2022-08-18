from linearlayer import LinearLayer


class InputLayer(LinearLayer):
    def __init__(self, image_shape, next_layer, **kwargs):
        self.__image_shape = image_shape
        super().__init__(size=self.linear_size, next_layer=next_layer, **kwargs)
    
    @property
    def input_shape(self):
        return self.__image_shape

    @property
    def output_shape(self):
        return self.__image_shape
        
    @property
    def values(self):
        return super().values
    
    @values.setter
    def values(self, new_values):
        assert len(new_values.shape) == 4  # images: count x channels x height x width, height == width
        self._values = self.activation(new_values)

    def back_propagation(self):
        return

    def _create_mask(self):
        return 1

