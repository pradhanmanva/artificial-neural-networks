import random

from neural_network import *


def get_data_from_csv():
    pass


def test(num_i, num_h):
    network = Network()
    input_nodes = [input_node(i) for i in range(num_i)]
    hidden_nodes = [Node() for i in range(num_h)]
    output_node = Node()

    # weights are all randomized
    for inode in input_nodes:
        for node in hidden_nodes:
            Edge(inode, node)

    for node in hidden_nodes:
        Edge(node, output_node)

    network.output_node = output_node
    network.input_node.extend(input_nodes)

    # labeled_examples = get_data_from_csv()
    labeled_examples = [((0, 0, 0), 1),
                        ((0, 0, 1), 0),
                        ((0, 1, 0), 1),
                        ((0, 1, 1), 0),
                        ((1, 0, 0), 1),
                        ((1, 0, 1), 0),
                        ((1, 1, 0), 1),
                        ((1, 1, 1), 0)]
    network.train(labeled_examples, max_iterations=5000)

    # test for consistency
    for number, isEven in labeled_examples:
        print("Error for %r is %0.4f. Output was:%0.4f" % (
            number, isEven - network.evaluate(number), network.evaluate(number)))


def make_network(num_i, num_h, num_each_layer):
    network = Network()
    input_nodes = [input_node(i) for i in range(num_i)]
    output_node = Node()
    network.output_node = output_node
    network.input_node.extend(input_nodes)

    layers = [[Node() for _ in range(num_each_layer)] for _ in range(num_h)]

    # weights are all randomized
    for inode in input_nodes:
        for node in layers[0]:
            Edge(inode, node)

    for layer1, layer2 in [(layers[i], layers[i + 1]) for i in range(num_h - 1)]:
        for node1 in layer1:
            for node2 in layer2:
                Edge(node1, node2)

    for node in layers[-1]:
        Edge(node, output_node)

    return network


def big_dataTest():
    network = make_network(2, 2, 15)

    big_data = []

    with open('clean.csv', 'r') as dataFile:
        for line in dataFile:
            (exampleStr, classStr) = line.split(',')
            big_data.append(([int(x) for x in exampleStr.split()], float(classStr)))

    random.shuffle(big_data)
    trainingData, testData = big_data[:-500], big_data[-500:]

    network.train(trainingData, learning_rate=0.5, max_iterations=10000)
    errors = [abs(testPt[-1] - round(network.evaluate(testPt[0]))) for testPt in testData]
    print("Average error: %.4f" % (sum(errors) * 1.0 / len(errors)))


if __name__ == "__main__":
    test(2, 4)
    big_dataTest()
