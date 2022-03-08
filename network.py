from random import random
import gzip

LITE_MODE = False

def sigmoid(x):
    return 0.5 * (x / (1 + abs(x))) + 0.5

# Hack for array representation
class IHaveNoQuotes:
    def __init__(self, string):
        self.intern = string

    def __str__(self):
        return self.intern

    def __repr__(self):
        return self.intern

class Node:
    def __init__(self):
        self.sources = []
        self.weights = []
        self.bias = 2 * random() - 1
        self.value = 0

    def compute(self):
        self.value = 0
        for i in range(len(self.sources)): # iterate and apply the weights to connections
            self.value += self.sources[i].value * self.weights[i]
        # apply the bias
        self.value -= self.bias
        # sigmoid value to clamp between 0 and 1
        self.value = sigmoid(self.value)
        pass

    def _rep(self):
        internals = [IHaveNoQuotes("(" + str(self.sources[i]) + "," + str(self.weights[i]) + ")") for i in range(len(self.sources))]
        i_str = str(internals)
        return "Node[" + str(len(self.sources)) + "]" + ("" if LITE_MODE else "{" + i_str[1:len(i_str)-1] + ",bias=" + str(self.bias) + "}")

    def __str__(self):
        return self._rep()

    def __repr__(self):
        return self._rep()

def generateLayer(previous_nodes, n):
    next_nodes = []
    for i in range(n):
        node = Node()
        node.sources = previous_nodes
        node.weights = [4 * random() - 2 for k in node.sources]
        next_nodes.append(node)
    return next_nodes

def mutate(weights, large_mutation = False):
    freq = 0.08 if large_mutation else 0.03
    
    for i in range(len(weights)):
        if random() < freq:
            weights[i] += (random() - 0.5) * (3 if large_mutation else 1)

# network lol
class Network:
    def __init__(self, inputs):
        self._vlayers = [inputs] # view layers
        self._layers = [] # actual step layers for calc
        self.first = inputs
        self.last = inputs
        self.cost = "Something has gone wrong!"

    def push(self, n):
        self.last = generateLayer(self.last, n)
        self._layers.append(self.last)
        self._vlayers.append(self.last)

    def compute(self, values=None):
        # initialise values
        if values is not None:
            for i in range(len(values)):
                self.first[i].value = values[i]

        return self._compute()

    def _compute(self):
        # perform steps
        for step in self._layers:
            for node in step:
                node.compute()

        return [node.value for node in self.last]

    def clone(self):
        result = Network(self.first)

        for step in self._layers:
            result.push(len(step))
            # copy weights and bias
            for i in range(len(step)):
                result.last[i].weights = step[i].weights.copy()
                result.last[i].bias = step[i].bias

        return result

    """
    Like clone, but mutates some values slightly.
    """
    def reproduce(self):
        result = Network(self.first)
        large_mutation = random() < 0.1

        for step in self._layers:
            result.push(len(step))
            # copy weights and bias
            for i in range(len(step)):
                result.last[i].weights = step[i].weights.copy()
                mutate(result.last[i].weights, large_mutation)
                result.last[i].bias = step[i].bias + (random() - 0.5) * (0.125 if large_mutation else 0.02)

        return result

    def __str__(self):
        return str(self._vlayers)
    def __repr__(self):
        return str(self._vlayers)

if __name__ == "__main__":
    # test serialisation
    input_1 = Node()
    input_2 = Node()
    nwk = Network([input_1, input_2])
    nwk.push(3)
    nwk.push(1)

    print(nwk)
    print(nwk.compute([0, 1]))

    with gzip.open("file.nwk", "wt") as f:
        f.write(str(nwk))

    with gzip.open("file.nwk", "rt") as f:
        print(f.read())

    
