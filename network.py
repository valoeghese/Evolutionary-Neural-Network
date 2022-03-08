"""
Evolutionary Neural Network in Python
Because python is a lazy language and also because I can't be bothered to do 100000 dimensional calculus
Maybe later I'll make it learn properly
"""

## TODO ##
# Store state into a file using repr (temp set long regardless if so) or another system and read the state.
# Ability to train. Original network and a series of mutations compete in identifying the set of colours. Strongest wins. Use output values diff sum thing for cost.
##########

from random import random, choice
import json

SHORT_REP = True # whether to strip internals for node string/repr and only show the number of sources
COMPETING = 6 # 6 compete at a time

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
        return "Node[" + str(len(self.sources)) + "]" + ("" if SHORT_REP else "{" + i_str[1:len(i_str)-1] + ",bias=" + self.bias + "}")

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

red_channel = Node()
green_channel = Node()
blue_channel = Node()

"""
network = Network([red_channel, green_channel, blue_channel])

network.push(16) # Intermediate Stage for complexity
network.push(11) # Colours to pick from ROYGB, Purple, Pink, Grey, Brown, White, Black

network2 = network.clone()
network3 = network.reproduce()
    
print(network.compute([0, 1, 0.5]))
print("")
print(network.compute([0.433, 0.2, 0]))
print("")
print(network.compute([0, 1, 0.5])) # consistency test
print("")
print(network2.compute([0.433, 0.2, 0])) # clone test
print("")
print(network3.compute([0.433, 0.2, 0])) # reproduce test

red_channel.value = 0
green_channel.value = 1
blue_channel.value = 0.5
print("\ncompute():")
print(network.compute()) # consistency test with manual set-for-all
print("")
print(network2.compute()) # consistency test with manual set-for-all
"""

current_networks = []

with open("nndata.json", "r") as file:
    _qwddf = json.load(file) # temp var
    correct_answers = _qwddf["entries"]
    node_index_map = _qwddf["translations"]

for i in range(COMPETING):
    _network = Network([red_channel, green_channel, blue_channel])

    _network.push(16) # Intermediate Stage for complexity
    _network.push(11) # Colours to pick from ROYGB, Purple, Pink, Grey, Brown, White, Black
    current_networks.append(_network)

cmd = input("[T] Train, [A] Apply> ")

def nodeMap(index):
    result = [0.0] * len(node_index_map)
    result[node_index_map[index]] = 1.0
    return result

"""
Gets the key of a value in a dictionary.
"""
def keyOf(dict_, val):
    keys = list(dict_.keys())
    vals = list(dict_.values())
    return keys[vals.index(val)]

def vecAbsSub(a, b):
    return [abs(a_i - b_i) for a_i, b_i in zip(a, b)]

""" more debug code
print(choice(correct_answers))
print(choice(correct_answers))
print(choice(correct_answers))
print(choice(correct_answers))
print(choice(correct_answers))
"""

def trainStep():
    global current_networks
    test_item = choice(correct_answers)
    
    # set the values
    red_channel.value = test_item["in"][0]
    green_channel.value = test_item["in"][1]
    blue_channel.value = test_item["in"][2]
    
    # get expected answer in terms of the node value map
    expected = nodeMap(test_item["out"])
    
    # test the networks and find the best (lowest) cost network
    best_network = None
    best_cost = 999999999 # big number
    
    for network in current_networks:
        # compute the result
        computed_result = network.compute()
        # compute cost
        computed_cost = sum(vecAbsSub(computed_result, expected))
        if computed_cost < best_cost:
            best_cost = computed_cost
            best_network = network
            
    # let the best network reproduce
    #print("Best Cost " + str(best_cost))
    current_networks = [best_network]

    for i in range(COMPETING - 2):
        current_networks.append(best_network.reproduce())

    # add in another random network
    _network = Network([red_channel, green_channel, blue_channel])

    _network.push(16)
    _network.push(11)
    current_networks.append(_network)

if "T" == cmd:
    # train 5 million cycles
    for i in range(5_000_000):
        trainStep()
    # next, apply
    cmd = "A"

if "A" == cmd:
    # apply
    while True:
        cmd = input("> ")
        if "exit" == cmd:
            break
        computed_result = current_networks[0].compute([float(a.strip()) for a in cmd.split(",")])
        result_index = computed_result.index(max(computed_result)) # get the index of the one with the max value
        print(keyOf(node_index_map, result_index)) # get the human readable form of that number



