import numpy as np



class Weights:
    def __init__(self, values: np.ndarray = None, shape: tuple = None, scale_coef: float = 1.0):
        self.__scale_coef = scale_coef
        if (values is None):
            self.__values = np.zeros(shape=shape)
            self.set_random_weights()
        else:
            self.__values = values

    @property
    def scale_coef(self):
        return self.__scale_coef

    @scale_coef.setter
    def scale_coef(self, new_scale_coef):
        self.unscale()
        self.__scale_coef = new_scale_coef
        self.scale()

    @property
    def shape(self):
        return self.values.shape
    
    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, new_values):
        assert new_values.shape == self.shape
        self.__values = np.copy(new_values)

    def set_random_weights(self):
        self.values = np.random.rand(*self.shape)
        self.scale()

    def scale(self):
        self.values = (2 * self.values - 1) * self.scale_coef

    def unscale(self):
        self.values = self.values / (2 * self.scale_coef) + 0.5