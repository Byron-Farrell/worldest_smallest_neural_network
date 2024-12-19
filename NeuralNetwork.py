class NeuralNetwork:

    def __init__(self):
        self.weight = 10
        self.bias = 5
        self.learning_rate = 0.01

    def feed_forward(self, x):
        w = self.weight
        b = self.bias

        return (x * w) + b


    def cost_derivative(self, o, x):
        return  2 * (o - x)


    def backprop(self, input, expected_output):

        predicted_output = self.feed_forward(input)

        Dc_Dx = self.cost_derivative(predicted_output, expected_output)

        Df_Dw = Dc_Dx * input
        Df_Db = Dc_Dx * 1

        self.weight = self.weight - (self.learning_rate * Df_Dw)
        self.bias = self.bias - (self.learning_rate * Df_Db)



if __name__ == "__main__":
    import numpy

    net = NeuralNetwork()
    inputs = numpy.random.uniform(low=1, high=10, size=(10,))
    inputs = [int(x) for x in inputs]
    outputs = [(2 * x) + 1 for x in inputs]

    for x, y in zip(inputs, outputs):
        net.backprop(x, y)

    test_inputs =  [1,2,3,4,5,6,7,8,9,10]
    test_outputs = [3,5,7,9,11,13,15,17,19,21]

    print("Predicted \t|  Expected \t|  Accuracy")
    print("------------------------------------------")
    for x, y in zip(test_inputs, test_outputs):
        t = net.feed_forward(x)
        p = y / 100
        f = (y - t)
        if f < 0:
            f = f * -1
        f = y - f
        p = f / p

        print("{:.2f} \t\t|  {:.2f} \t\t|  ({:.2f}%)".format(t, y, p))