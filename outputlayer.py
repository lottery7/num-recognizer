from linearlayer import LinearLayer


class OutputLayer(LinearLayer):
    def __init__(self, size, prev_layer, **kwargs):
        super().__init__(size=size, prev_layer=prev_layer, **kwargs)

    def _create_mask(self):
        return 1