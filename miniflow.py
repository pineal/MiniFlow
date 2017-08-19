import numpy as np

def topological_sort(feed_dict):
    """
    - Sort generic nodes in topoligical order using Kahn's Algorithm
    - 'feed_dict': a dictionary where
        - the key is a 'Input' node and 
        - the value is the respective value feed to that node.
    - Return a list of sorted nodes.
    """
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in':set(), 'out', set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]
        
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['output'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L





def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Args:
        'output_node': A node in the graph, should be the output node (have no outgoing edges)
        'sorted_nodes': a topologically sorted list of nodes

    Return:
        the output Node's value
    """
    for n in sorted_nodes:
        n.forward()

    return output_node.value





"""
Base class: defines the base set of properties taht every node holds

Two lists are stored in node:
- One to store references to the inbound nodes
- The other to store references to the outbound nodes
"""
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound Node here, andd this Node as an outbound Node to _that_ Node.
        for node in inbound_nodes:
            node.outbound_nodes.append(self)
        # Result, a calculated value 
        self.value = None

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on 'inbound_nodes' and 
        store the result in self.value
        """
        raise NotImplementedError

"""
class Input: Subclass of Node
- Input subclass does not actually calculate anything.
- The input subclass just holds a value, such as a data feature
  or a model parameter(wight/bias).
- You can set value either explicitly or with the forward() method
"""
class Input(Node):
    def __init__(self):
        # As Input node has no inbound nodes
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)
    
    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward()
    # 
    # All other node implementations should get the value
    # of the previous node from self.inbound_bounds
    #
    # Example:
    # va10 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

"""
class Add: Subclass of Node 
perform a calculation - addition
"""

class Add(Node):
    def __init__(self, *inputs):
        node.__init__(self, inputs)

    def forward(self):
        self.value = 0
        for i in range(0, len(self.inbound_nodes)):
            self.value = self.value + self.inbound_nodes[i].value


"""
class Linear:
"""

class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b]):

    def forward(self):
        """
        Set self.value to the value of the linear function output.
        """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b

"""
class Sigmoid: 
    Conceptually the sigmoid function makes decisions.
    When given weighted features from some data, it indicates whether or not
    the features contribure to a classfication.
    In that way, a sigmoid activation works well forwarding a linear transformation.
"""
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        Will be used in both 'forward' and 'backward'
        Args:
            x: A numpy array-like object
        Return:
            result of sigmoid function - apply the step function
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inbound_nodes[0].value)

"""
class MSE:
    squarded error cost function
    should be used as the last node for a network
"""

class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])


    def forward(self):
    """
    Calculated the mean squared error.
    """
    # NOTE: We reshape these to avoid possible matrix/vector broadcast errors.
    # For example: if we subtract an array of shape (3,) from an array of shape (3, 1)
    # we get an array of shape(3,3) as the result when we want an array of shape (3, 1) instead
    #
    # Making both arrays (3,1) insures the result is (3,1) and does
    # an elementwise subtraction as expeceted.

    y = self.inbound_nodes[0].value.reshape(-1, 1)
    a = self.inbound_nodes[1].value.reshape(-1, 1)
    m = self.inbound_node[0].value.shape[0]

    self.value = np.mean((y - a)**2)



