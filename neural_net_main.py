import sys

import pandas as pd

from neural_network import *
from reader import *


def dummy_test(num_i, num_h):
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
    network.train(labeled_examples, max_iterations=50)

    # test for consistency
    for number, isEven in labeled_examples:
        print("Error for %r is %0.4f. Output was:%0.4f" % (
            number, isEven - network.evaluate(number), network.evaluate(number)))


def make_network(path, num_h, num_each_layer):
    df = pd.read_csv(path, header=None)
    num_i = len(df.columns)

    network = Network()
    input_nodes = [input_node(i) for i in range(num_i)]
    output_node = Node()
    network.output_node = output_node
    network.input_node.extend(input_nodes)

    layers = []
    for nodes in num_each_layer:
        layer = []
        for i in range(nodes):
            layer.append(Node())
        layers.append(layer)

    # weights are all randomized
    for inode in input_nodes:
        for node_l in layers[0]:
            Edge(inode, node_l)

    for layer1, layer2 in [(layers[i], layers[i + 1]) for i in range(num_h - 1)]:
        n = 0
        for node1 in layer1:
            for node2 in layer2:
                Edge(node1, node2)

    for node in layers[-1]:
        Edge(node, output_node)

    return network


def big_data_test(path, percent, epoch, network, rate):
    big_data = change_csv('clean.csv')

    random.shuffle(big_data)
    df = pd.read_csv(path, header=None)
    rows = int(len(df.index) * (percent / 100))
    training_data, test_data = big_data[:rows:-1], big_data

    errors = []
    network.train(training_data, rate, epoch)
    for train_pt in training_data:
        error = abs(train_pt[-1] - network.evaluate(train_pt[0]))
        errors.append(error)
    print("\nTraining Average error: %.4f" % (sum(errors) * 1.0 / len(errors)))

    errors = []
    for test_pt in test_data:
        error = abs(test_pt[-1] - network.evaluate(test_pt[0]))
        errors.append(error)
    print("Test Average error: %.4f" % (sum(errors) * 1.0 / len(errors)))


def print_network(path, network, hidden_layers):
    df = pd.read_csv(path, header=None)
    num_i = len(df.columns)

    print("Input Layer:", end=" ")
    i = 0
    for node in network.input_node:
        print("\nNeuron " + str(i + 1) + " Weights:", end=" ")
        for e in node.outgoing_edges:
            print(" %.4f" % e.weight, end=" ")
        i += 1

    num_layers = 0
    for i in hidden_nodes:
        x = network.input_node[i]
        print("\nLayer " + str(num_layers + 1), end=" ")
        num_hnodes = 0
        for j in range(i+1):
            y = x.outgoing_edges[0].weight
            print("\nNeuron : " + str(num_hnodes + 1), end=" ")
            for k in range(j+1):
                print(" %.5f " % network.return_weights(), end=" ")
            num_hnodes += 1
        num_layers += 1

if __name__ == "__main__":
    # dummy_test(2, 4)
    cmd_line = sys.argv
    output_path = cmd_line[1]
    training_percent = int(cmd_line[2])
    iteration = int(cmd_line[3])
    hidden_layers = int(cmd_line[4])
    i = 5
    hidden_nodes = []
    while i < len(cmd_line):
        hidden_nodes.append(int(cmd_line[i]))
        i += 1
    learning_rate = 0.9
    network = make_network(output_path, hidden_layers, hidden_nodes)
    print_network(output_path, network, hidden_layers)
    big_data_test(output_path, training_percent, iteration, network, learning_rate)
