from typing import Tuple
import numpy as np
from keras.datasets import mnist
from layers import *
from activationfunc import *



class NumRecognizer:
    LEARN_DATA_SIZE = 1000
    INPUT_IMAGE_CHANNELS = 1
    INPUT_IMAGE_SIZE = 28  # 28 x 28 pixels
    INPUT_SIZE = INPUT_IMAGE_SIZE**2
    OUTPUT_SIZE = 10
    BATCH_SIZE = 100
    ALPHA = 0.1
    NUM_ITERS = 10

    def __init__(self):
        self.__layers = [
            InputLayer((self.INPUT_IMAGE_CHANNELS, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE), None),
            OutputLayer(self.OUTPUT_SIZE, None)
        ]
        self.__layers[1].prev_layer = self.__layers[0]

        self.__init_dataset()
    
    @property
    def layers(self) -> list:
        return self.__layers

    def __init_dataset(self) -> None:
        (learn_inputs, learn_outputs), (test_inputs, test_outputs) = mnist.load_data()

        self.__learn_inputs = (learn_inputs[:self.LEARN_DATA_SIZE] / 255)[:, np.newaxis, :, :]
        self.__test_inputs = (test_inputs / 255)[:, np.newaxis, :, :]

        self.__learn_outputs = np.zeros(shape=(self.LEARN_DATA_SIZE, self.OUTPUT_SIZE))
        for ind, out in enumerate(learn_outputs[:self.LEARN_DATA_SIZE]):
            self.__learn_outputs[ind][out] = 1.0

        self.__test_outputs = np.zeros(shape=(len(test_outputs), self.OUTPUT_SIZE))
        for ind, out in enumerate(test_outputs):
            self.__test_outputs[ind][out] = 1.0

    def __learn_batch_of_inputs(self, inputs:np.ndarray, expected_outputs:np.ndarray, batch_size:int) -> int:
        inputs = inputs.reshape(batch_size, self.INPUT_IMAGE_CHANNELS, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE)
        self.produce_forward(inputs, mask=True)
        outputs = self.layers[-1].values
        correct_counter = 0

        for k in range(batch_size):
            correct_counter += (
                np.argmax(outputs[k]) == np.argmax(expected_outputs[k])
            )

        delta = (outputs - expected_outputs) / batch_size
        self.produce_backward(delta)

        for layer in self.layers[:0:-1]:
            layer.update_weights(self.ALPHA)

        return correct_counter
        
    def learn(self) -> None:
        ans = 1
        i = 0
        while (ans > 0):
            for _ in range(ans):
                i += 1
                correct_counter = 0
                for j in range(self.LEARN_DATA_SIZE // self.BATCH_SIZE):
                    batch_start, batch_end = j * self.BATCH_SIZE, (j + 1) * self.BATCH_SIZE

                    inputs = self.__learn_inputs[batch_start:batch_end]
                    expected_outputs = self.__learn_outputs[batch_start:batch_end]
                    correct_counter += self.__learn_batch_of_inputs(inputs, expected_outputs, self.BATCH_SIZE)
                
                if (i % 1 == 0):
                    print(
                        f"Iter: {i}",
                        f"Train-Acc: {correct_counter / self.LEARN_DATA_SIZE * 100:.1f}%",
                        # end=' ' if i % 100 == 0 else '\n'
                    )
            
            ans = input()
            if (ans == "test"):
                self.test()
                ans = int(input())
            else:
                ans = int(ans)



    def test(self) -> None:
        correct_counter = 0
        for i in range(len(self.__test_inputs) // self.BATCH_SIZE):
            batch_start, batch_end = i * self.BATCH_SIZE, (i + 1) * self.BATCH_SIZE

            inputs = self.__test_inputs[batch_start:batch_end]
            expected_outputs = self.__test_outputs[batch_start:batch_end]
            outputs = self.get_answer(inputs)

            for k in range(self.BATCH_SIZE):
                correct_counter += (
                    np.argmax(outputs[k]) == np.argmax(expected_outputs[k])
                )
        print(f"Test-Acc: {correct_counter / len(self.__test_inputs) * 100:.1f}%", end='\t')

        correct_counter = 0
        for i in range(self.LEARN_DATA_SIZE // self.BATCH_SIZE):
            batch_start, batch_end = i * self.BATCH_SIZE, (i + 1) * self.BATCH_SIZE

            inputs = self.__learn_inputs[batch_start:batch_end]
            expected_outputs = self.__learn_outputs[batch_start:batch_end]
            outputs = self.get_answer(inputs)

            for k in range(self.BATCH_SIZE):
                correct_counter += (
                    np.argmax(outputs[k]) == np.argmax(expected_outputs[k])
                )
        print(f"Learn-Acc: {correct_counter / self.LEARN_DATA_SIZE * 100:.1f}%")


    def get_answer(self, inputs:np.ndarray) -> np.ndarray:
        self.produce_forward(inputs)
        return self.layers[-1].values
    
    def produce_forward(self, inputs:np.ndarray, mask=False) -> None:
        self.layers[0].values = inputs
        for i, layer in enumerate(self.layers[1:]):
            layer.update_values(mask)
    
    def produce_backward(self, delta:np.ndarray) -> None:
        self.layers[-1].delta_coef = delta
        for layer in self.layers[:1:-1]:
            layer.back_propagation()
        
    def add_linear_layer(self, size:int, activation:tuple=None) -> None:
        layer = LinearLayer(
            size,
            activation=activation or (relu, relu_deriv),
            prev_layer=self.layers[-2],
            next_layer=self.layers[-1]
        )
        self.layers.insert(-1, layer)

    def add_conv_layer(self, kernel_size:int, kernel_output_size:int, padding=0, stride=1, activation:tuple=None) -> None:
        layer = ConvLayer(
            kernel_size,
            kernel_output_size,
            padding=padding,
            stride=stride,
            activation=activation or (relu, relu_deriv),
            prev_layer=self.layers[-2],
            next_layer=self.layers[-1]
        )
        self.layers.insert(-1, layer)

    def add_average_pooling(self, filter_size:int, stride:int) -> None:
        layer = AveragePooling(
            filter_size=filter_size,
            stride=stride,
            prev_layer=self.layers[-2],
            next_layer=self.layers[-1],
        )
        self.layers.insert(-1, layer)
        

if __name__ == "__main__":
    nn = NumRecognizer()
    nn.add_conv_layer(5, 6, padding=2, stride=1, activation=(relu, relu_deriv))
    nn.add_average_pooling(2, 2)
    nn.add_conv_layer(5, 3, padding=0, stride=1, activation=(relu, relu_deriv))
    nn.add_average_pooling(2, 2)
    nn.add_linear_layer(512, activation=(relu, relu_deriv))
    nn.learn()
    
    