import numpy
random_number_generator = numpy.random.default_rng()

class NeuralNetwork:

    def __init__(self):
        self.weight = random_number_generator.random() * 10
        self.bias = random_number_generator.random() * 5
        self.learning_rate = 0.01

    def feed_forward(self, x):
        w = self.weight
        b = self.bias

        return (x * w) + b


    def cost_derivative(self, y, y_hat):
        return  2 * (y - y_hat)


    def backprop(self, input, expected_output):

        predicted_output = self.feed_forward(input)

        partialC_partialX = self.cost_derivative(predicted_output, expected_output)

        partialF_partialW = input
        partialF_partialB = 1

        w_gradient = partialC_partialX * partialF_partialW
        b_gradient = partialC_partialX * partialF_partialB

        delta_w = self.learning_rate * w_gradient
        delta_b = self.learning_rate * b_gradient

        self.weight = self.weight - delta_w
        self.bias = self.bias - delta_b

    def evaluation(self, test_inputs, test_outputs):
        print("Predicted \t|  Expected \t|  Accuracy")
        print("------------------------------------------")
        for x, y in zip(test_inputs, test_outputs):
            predicted = self.feed_forward(x)
            error = (y - predicted)

            if error < 0:
                error *= -1

            evaluation = (y - error) / (y / 100)

            print("{:.2f} \t\t|  {:.2f} \t\t|  ({:.2f}%)".format(predicted, y, evaluation))

if __name__ == "__main__":
    net = NeuralNetwork()
    inputs = random_number_generator.random((100,)) * 10
    outputs = [(2 * x) + 1 for x in inputs]

    for x, y in zip(inputs, outputs):
        net.backprop(x, y)

    test_inputs =  [x for x in range(1, 11)]
    test_outputs = [(2 * x) + 1 for x in test_inputs]

    net.evaluation(test_inputs, test_outputs)
