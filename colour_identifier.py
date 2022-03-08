"""
Evolutionary Neural Network in Python
Because python is a lazy language and also because I can't be bothered to do 100000 dimensional calculus
Maybe later I'll make it learn properly
"""

## TODO ##
# Store state into a file using repr (temp set long regardless if so) or another system and read the state.
##########

from network import *
import json

def vecAbsSub(a, b):
    return [abs(a_i - b_i) for a_i, b_i in zip(a, b)]

class SumAbsDifference:
    def compute(self, computed, expected):
        return sum(vecAbsSub(computed, expected))

class PunishAndReward:
    def compute(self, computed, expected):
        result = 0
        for i in range(len(computed)): # over every index
            if (computed[i] > 0.5) == (expected[i] > 0.5): # reward if the same direction of 0.5 as the expected value
                result -= 1
            else: # otherwise punish
                result += 1
        return result

##################
##  Parameters  ##
##################

COMPETING = 7 # the number of networks competing at a time. One will be the original, one a completely random network (useful early on to prevent homogeniety) and the rest will be mutations of the original.

COST_FUNCTION = PunishAndReward() # use this cost function

CYCLES = 60_000 # have this many training cycles
REPORT_EVERY = 5_000 # report expected avg cost every this many cycles.

red_channel = Node()
green_channel = Node()
blue_channel = Node()

def generateNewNetwork():
    result = Network([red_channel, green_channel, blue_channel])

    result.push(16) # Intermediate Stage for complexity
    result.push(16) # Intermediate Stage 2 for complexity
    result.push(11) # Colours to pick from ROYGB, Purple, Pink, Grey, Brown, White, Black
    return result

current_networks = []

with open("nndata.json", "r") as file:
    _qwddf = json.load(file) # temp var
    correct_answers = _qwddf["entries"]
    node_index_map = _qwddf["translations"]

for i in range(COMPETING):
    _network = generateNewNetwork()
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

pass_n = 0
passes_per_training_session = len(correct_answers)

def trainNetworksOn(test_item):
    # set the values
    red_channel.value = test_item["in"][0]
    green_channel.value = test_item["in"][1]
    blue_channel.value = test_item["in"][2]

    # get expected answer in terms of the node value map
    expected = nodeMap(test_item["out"])

    for network in current_networks:
        # compute the result for this pass
        computed_result = network.compute()
        # compute cost
        network.cost += COST_FUNCTION.compute(computed_result, expected)

def trainStep():
    global current_networks, pass_n
    pass_n += 1

    # reset costs
    for network in current_networks:
        network.cost = 0

    # iterate passes

    for test_item in correct_answers:
        trainNetworksOn(test_item)
    
    # test the networks and find the best (lowest) cost network
    best_network = None
    best_cost = 999999999 # big number
    
    for network in current_networks:
        if network.cost < best_cost:
            best_cost = network.cost
            best_network = network
            
    # let the best network reproduce
    #print("Best Cost " + str(best_cost))
    current_networks = [best_network] # it will compete against its children to assure the stats don't get worse

    for i in range(COMPETING - 2):
        current_networks.append(best_network.reproduce())

    # add in another random network
    _network = generateNewNetwork()
    current_networks.append(_network)

    if pass_n % REPORT_EVERY == 0:
        print("Progress Report: network of generation " + str(pass_n) + " has expected ~average cost " + str(current_networks[0].cost / passes_per_training_session))

if "T" == cmd:
    trainStep()
    
    print("Best initial network has expected ~average cost " + str(current_networks[0].cost / passes_per_training_session))
    
    for i in range(CYCLES - 1):
        trainStep()
    # say the cost of the result network
    print("Achieved network of expected ~average cost " + str(current_networks[0].cost / passes_per_training_session))
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
        print(keyOf(node_index_map, result_index), end=" ") # get the human readable form of that number
        print(computed_result) # can be interesting to see this



