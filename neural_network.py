import math
import random


def activation_function(x):
    '''
    activation function : sigmoid
    :param x: input x
    :return: get 1/(1+e^(-x))
    '''
    return 1.0 / (1.0 + math.exp(-x))


class Node:
    '''
    making nodes of the network
    '''

    def __init__(self):
        '''
        initialising the attributes of the node
        '''
        self.last_output = None
        self.last_input = None
        self.error = None
        self.outgoing_edges = []
        self.incoming_edges = []
        self.add_bias()

    def add_bias(self):
        '''
        appending the bias node to the incoming edges
        '''
        self.incoming_edges.append(Edge(bias_node(), self))

    def evaluate(self, input_vector):
        '''
        :param input_vector: the vector containing the input to the node
        :return: last_output from the node to the next node
        '''
        if self.last_output is not None:
            return self.last_output

        self.last_input = []
        weighted_sum = 0

        for e in self.incoming_edges:
            input_i = e.source.evaluate(input_vector)
            self.last_input.append(input_i)
            weighted_sum += e.weight * input_i

        self.last_output = activation_function(weighted_sum)
        self.evaluate_cache = self.last_output
        return self.last_output

    def get_error(self, label):
        '''
        Get the error for a given node in the network. If the node is an
           output node, label will be used to compute the error. For an input node, we
           simply ignore the error.
        :param label: to check the label of the predicted class
        :return: return the error
        '''

        if self.error is not None:
            return self.error
        assert self.last_output is not None
        if not self.outgoing_edges:  # this is an output node
            self.error = label - self.last_output
        else:
            self.error = sum([edge.weight * edge.target.get_error(label) for edge in self.outgoing_edges])
        return self.error

    def update_weights(self, learning_rate):
        '''
        Update the weights of a node, and all of its successor nodes.
           Assume self is not an input_node. If the error, last_output, and
           lastInput are None, then this node has already been updated.
        :param learning_rate:
        :return: nothing
        '''

        if self.error is not None and self.last_output is not None and self.last_input is not None:
            for i, edge in enumerate(self.incoming_edges):
                edge.weight += (learning_rate * self.last_output * (1 - self.last_output) *
                                self.error * self.last_input[i])

            for edge in self.outgoing_edges:
                edge.target.update_weights(learning_rate)

            self.error = None
            self.last_input = None
            self.last_output = None

    def clear_evaluate_cache(self):
        if self.last_output is not None:
            self.last_output = None
            for edge in self.incoming_edges:
                edge.source.clear_evaluate_cache()


class input_node(Node):
    ''' Input nodes simply evaluate to the value of the input for that index.
     As such, each input node must specify an index. We allow multiple copies
     of an input node with the same index (why not?). '''

    def __init__(self, index):
        Node.__init__(self)
        self.index = index

    def evaluate(self, input_vector):
        self.last_output = input_vector[self.index]
        return self.last_output

    def update_weights(self, learning_rate):
        for edge in self.outgoing_edges:
            edge.target.update_weights(learning_rate)

    def get_error(self, label):
        for edge in self.outgoing_edges:
            edge.target.get_error(label)

    def add_bias(self):
        pass

    def clear_evaluate_cache(self):
        self.last_output = None


class bias_node(input_node):
    def __init__(self):
        Node.__init__(self)

    def evaluate(self, input_vector):
        return 1.0


class Edge:
    '''
    The class Edge that connects the nodes to each other. Here the weights are randomly assigned to them.
    '''

    def __init__(self, source, target):
        self.weight = random.uniform(0, 1)  # assigning random weight
        self.source = source  # type:Node
        self.target = target  # type:Node

        # attach the edges to its nodes
        source.outgoing_edges.append(self)
        target.incoming_edges.append(self)


class Network:
    def __init__(self):
        self.input_node = []
        self.output_node = None

    def evaluate(self, input_vector):
        assert max([v.index for v in self.input_node]) < len(input_vector)
        self.output_node.clear_evaluate_cache()

        output = self.output_node.evaluate(input_vector)
        return output

    def propagate_error(self, label):
        for node in self.input_node:
            node.get_error(label)

    def return_weights(self):
        return random.uniform(0,1)

    def update_weights(self, learning_rate):
        '''
        Update the weights of the input nodes
        :param learning_rate: learning rate
        :return: none
        '''
        for node in self.input_node:
            node.update_weights(learning_rate)

    def train(self, labeled_examples, learning_rate=0.9, max_iterations=500):
        '''
        Used to train the labeled_examples on the network
        :param labeled_examples: training data to be used on the network
        :param learning_rate: learning rate at which the machine learns. If it is too low, the machine might take a long time to converge. If it is too small, the machine might never converge or might sk the point of convergence.
        :param max_iterations: number of iterations
        :return: none
        '''
        while max_iterations > 0:
            for example, label in labeled_examples:
                output = self.evaluate(example)
                self.propagate_error(label)
                self.update_weights(learning_rate)
                # print(output)
                max_iterations -= 1
