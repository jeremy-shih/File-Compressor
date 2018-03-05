"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# symbol_to_code functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for i in range(len(text)):
        if text[i] not in freq_dict:
            freq_dict[(text[i])] = 1
        else:
            freq_dict[(text[i])] += 1
    return freq_dict


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    >>> freq = {3: 6}
    >>> t = huffman_tree(freq)
    >>> result3 = HuffmanNode(None, HuffmanNode(3), None)
    >>> result6 = HuffmanNode(None, None, HuffmanNode(3))
    >>> t == result3 or t == result6
    True
    >>> freq = {1: 100, 2: 15, 3: 50}
    >>> t = huffman_tree(freq)
    >>> result4 = HuffmanNode(None, HuffmanNode(None, \
    HuffmanNode(2, None, None), HuffmanNode(3, None, None)), \
    HuffmanNode(1, None, None))
    >>> result5 = HuffmanNode(None, HuffmanNode(1, None, None), \
    HuffmanNode(None, HuffmanNode(2, None, None), HuffmanNode(3, None, None)))
    >>> t == result4 or t == result5
    True
    """
    # http://www.geeksforgeeks.org/greedy-algorithms-set-3-huffman-coding/

    # https://docs.python.org/2/library/heapq.html

    # create a list of of lists, with each inner
    # list being [frequency, symbol]
    freq_list = []
    for key in freq_dict:
        freq_list.append([freq_dict[key], key])
    freq_list.sort()

    # if the length of freq_list is only 1, that means we only need
    # one node for the huffman tree
    if len(freq_list) == 1:
        left = list(freq_dict.keys())[0]
        return HuffmanNode(None, HuffmanNode(left), None)
    # otherwise, use the helper to build the tree
    else:
        double_list = []

        for i in range(len(freq_list)):
            for key in freq_dict:
                if freq_dict[key] == freq_list[i][0] and key == freq_list[i][1]:
                    leaf = HuffmanNode(key)
                    double_list.append([freq_dict[key], leaf])
        double_list.sort()

        huffman_tree_helper(double_list)

        return double_list[0][1]


def huffman_tree_helper(lst):
    """
    Delete the first two nodes and their frequencies in lst, and
    append a new node from the sum of those two nodes.

    @param list of lists lst:
    @rtype: None
    """
    # take a list of lists with each inner list representing a "node"
    while len(lst) > 1:
        # delete two lowest nodes
        store = [lst[0], lst[1]]
        lst.remove(lst[0])
        lst.remove(lst[0])
        count = 0
        # find the sum
        for item in store:
            count += item[0]

        sum_list = [count, HuffmanNode(None, store[0][1], store[1][1])]
        # append the node of the sum to the double_list
        lst.append(sum_list)
        lst.sort()


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> left1 = HuffmanNode(None, HuffmanNode(1), HuffmanNode(3))
    >>> left = HuffmanNode(None, left1, HuffmanNode(2))
    >>> right1 = HuffmanNode(None, HuffmanNode(10), HuffmanNode(11))
    >>> right = HuffmanNode(None, HuffmanNode(9), right1)
    >>> tree = HuffmanNode(None, left, right)
    >>> a = get_codes(tree)
    >>> a == {1: '000', 3: '001', 2: '01', 9: '10', 10: '110', 11: '111'}
    True
    """
    # create an empty dictionary
    path_dict = {}
    # create a list of the symbols using a helper
    symbol_list = symbol_list_creator(tree)
    # for each symbol in that list, get its code and add it to the dict
    for symbol in symbol_list:
        p = symbol_to_code(tree, symbol)
        del p[-1]
        string = ''
        for num in p:
            string += str(num)
        path_dict[symbol] = string
    return path_dict


def symbol_list_creator(tree):
    """
    Create a list of symbols that the leaves of tree represent.

    @param HuffmanNode tree: tree for which to create a symbol list.
    @rtype: list

    >>> left1 = HuffmanNode(None, HuffmanNode(1), HuffmanNode(3))
    >>> left = HuffmanNode(None, left1, HuffmanNode(2))
    >>> right1 = HuffmanNode(None, HuffmanNode(10), HuffmanNode(11))
    >>> right = HuffmanNode(None, HuffmanNode(9), right1)
    >>> tree = HuffmanNode(None, left, right)
    >>> lst = symbol_list_creator(tree)
    >>> lst.sort()
    >>> lst
    [1, 2, 3, 9, 10, 11]
    """
    # given a huffman tree, create a list of all the symbols in the tree
    symbol_list = []
    if tree is None:
        return symbol_list
    elif tree.left is None and tree.right is None:
        symbol_list.append(tree.symbol)
        return symbol_list
    else:
        symbol_list = symbol_list_creator(tree.left) + \
                      symbol_list_creator(tree.right)
        return symbol_list


def symbol_to_code(t, symbol):
    """
    Return the code for symbol in t.

    @param HuffmanNode t: node to return code representation for
    @param object symbol: symbol to look for
    @rtype: list

    >>> left1 = HuffmanNode(None, HuffmanNode(1), HuffmanNode(3))
    >>> left = HuffmanNode(None, left1, HuffmanNode(2))
    >>> right1 = HuffmanNode(None, HuffmanNode(10), HuffmanNode(11))
    >>> right = HuffmanNode(None, HuffmanNode(9), right1)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> symbol_to_code(tree, 9)
    ['1', '0', 9]
    >>> d = {1: 100, 2: 15, 3: 50}
    >>> t = huffman_tree(d)
    >>> symbol_to_code(t, 1)
    ['1', 1]
    """
    # given a symbol, travel down the huffman tree, adding '0' or '1'
    # depending on whether you take the left or right path, respectively,
    # and then adding the symbol to the end
    if t is None:
        return []
    elif t.left is None and t.right is None:
        return [t.symbol] if t.symbol == symbol else []
    else:
        left = ['0'] + symbol_to_code(t.left, symbol)
        right = ['1'] + symbol_to_code(t.right, symbol)
        return left if left[-1] == symbol else right


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    >>> t = HuffmanNode(None, HuffmanNode(1), HuffmanNode(2))
    >>> number_nodes(t)
    >>> t.number
    0
    """

    queue = []

    def act1(t):
        """
        Append t to queue if t is an internal node.

        @param HuffmanNode t:
        @rtype: None
        """
        if t.left and t.right:
            queue.append(t)

    # postorder traverse the tree and append each internal node to queue
    postorder_visit(tree, act1)

    # because the list is in postorder, we can now simply assign a number
    # to each node based on its position in queue
    for j in range(len(queue)):
        queue[j].number = j


def postorder_visit(t, act):
    """
    Visit HuffmanNode t in postorder and act on nodes as you visit.

    @param HuffmanNode|None t: binary tree to visit
    @param (HuffmanNode)->Any act: function to use on nodes
    @rtype: None

    >>> b = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> def f(node): print(node.symbol)
    >>> postorder_visit(b, f)
    3
    2
    None
    """
    if t is not None:
        postorder_visit(t.left, act)
        postorder_visit(t.right, act)
        act(t)


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    codes_dict = get_codes(tree)
    total = 0
    total_freq = 0
    for key in codes_dict:
        # multiply the frequency of each symbol with the number of
        # bits needed to represent it
        product = len(codes_dict[key]) * freq_dict[key]
        total += product
        total_freq += freq_dict[key]
    return total / total_freq


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    >>> text = bytes([1, 2, 1, 0, 2, 2, 1, 2, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '11110111', '10000000']
    """
    string = ''
    lst = []

    for item in text:
        code = codes[item]
        # add code to an empty string if the length of string is less than 8
        if len(string + code) < 8:
            string += code
        # if you reach exactly 8 characters in string, append it to the list
        elif len(string + code) == 8:
            lst.append(string + code)
            string = ''
        # if you reach more than 8 characters, append the first 8 to the list
        # and set string as the leftover characters
        else:
            string += code
            lst.append(string[:8])
            string = string[8:]

    # fill in the last string with 0's if its length is less than 8
    if len(string) > 0:
        lst.append(string + '0' * (8 - len(string)))

    return bytes([bits_to_byte(bit) for bit in lst])


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> left = HuffmanNode(None, HuffmanNode(1), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(3), HuffmanNode(4))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 1, 0, 2, 0, 3, 0, 4, 1, 0, 1, 1]
    """
    lst = []

    def act(t):
        """ Append t to lst if t has a left child and a right child.

        @param HuffmanNode t:
        @rtype: None
        """
        if t.left is not None and t.right is not None:
            lst.append(tree_to_bytes_helper(t))

    # postorder traverse the tree and append the bytes representation of
    # each node to the list
    postorder_visit(tree, act)

    final = bytes([])
    for i in range(len(lst)):
        final += bytes(lst[i])

    return final


def tree_to_bytes_helper(tree):
    """
    Return a list of four bytes representing a single node.

    @param HuffmanNode tree: node to return a representation for
    @rtype: list of int

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes_helper(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(1), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(3), HuffmanNode(4))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes_helper(tree))
    [1, 0, 1, 1]
    >>> list(tree_to_bytes_helper(left))
    [0, 1, 0, 2]
    >>> list(tree_to_bytes_helper(right))
    [0, 3, 0, 4]
    """
    # return a bytes representation of one node
    if tree is None:
        return bytes([])
    elif not tree.left.left and not tree.right.right:
        return bytes([0]) + bytes([tree.left.symbol]) + bytes([0]) + \
               bytes([tree.right.symbol])
    elif not tree.left.left:
        return bytes([0]) + bytes([tree.left.symbol]) + bytes([1]) + \
               bytes([tree.right.number])
    elif not tree.right.right:
        return bytes([1]) + bytes([tree.left.number]) + bytes([0]) + \
               bytes([tree.right.symbol])
    else:
        return bytes([1]) + bytes([tree.left.number]) + bytes([1]) + \
               bytes([tree.right.number])


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), HuffmanNode(None, \
HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    >>> lst = [ReadNode(0, 1, 0, 2), ReadNode(0, 3, 0, 4), \
    ReadNode(1, 0, 0, 7), ReadNode(1, 1, 0, 12), ReadNode(1, 2, 1, 3)]
    >>> generate_tree_general(lst, 4)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(None, \
HuffmanNode(1, None, None), HuffmanNode(2, None, None)), \
HuffmanNode(7, None, None)), HuffmanNode(None, HuffmanNode(None, \
HuffmanNode(3, None, None), HuffmanNode(4, None, None)), \
HuffmanNode(12, None, None)))
    """
    # recursively generate a new tree based on correct byte representation of
    # each node.
    new_root = node_lst[root_index]
    if new_root.l_type == 0 and new_root.r_type == 0:
        return HuffmanNode(None, HuffmanNode(new_root.l_data),
                           HuffmanNode(new_root.r_data))
    elif new_root.l_type == 1 and new_root.r_type == 0:
        return HuffmanNode(None, generate_tree_general(node_lst,
                                                       new_root.l_data),
                           HuffmanNode(new_root.r_data))
    elif new_root.l_type == 0 and new_root.r_type == 1:
        return HuffmanNode(None, HuffmanNode(new_root.l_data),
                           generate_tree_general(node_lst, new_root.r_data))
    else:
        return HuffmanNode(None, generate_tree_general(node_lst,
                                                       new_root.l_data),
                           generate_tree_general(node_lst, new_root.r_data))


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)))
    >>> lst = [ReadNode(0, 5, 0, 6), ReadNode(1, 0, 0, 4), \
    ReadNode(0, 7, 0, 8), ReadNode(1, 0, 1, 0), ReadNode(1, 0, 0, 10), \
    ReadNode(1, 0, 0, 11)]
    >>> generate_tree_postorder(lst, 5)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(None, \
HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(6, None, None)), HuffmanNode(4, None, None)), \
HuffmanNode(None, HuffmanNode(7, None, None), \
HuffmanNode(8, None, None))), HuffmanNode(10, None, None)), \
HuffmanNode(11, None, None))
    >>> generate_tree_postorder(lst, 3)
    HuffmanNode(None, \
HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(6, None, None)), HuffmanNode(4, None, None)), \
HuffmanNode(None, HuffmanNode(7, None, None), \
HuffmanNode(8, None, None)))
    """
    overall = generate_overall_tree_postorder(node_lst)
    number_nodes(overall)
    lst = []

    def act(t):
        """ Append t to lst if t.number is equal to root_index.

        @param HuffmanNode t:
        @rtype: None
        """
        if t.number == root_index:
            lst.append(t)

    postorder_visit(overall, act)

    return lst[0]


def generate_overall_tree_postorder(node_lst):
    """
    Return the root of the Huffman tree corresponding
    to the last item of node_lst.

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_overall_tree_postorder(lst)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)))
    >>> lst = [ReadNode(0, 5, 0, 6), ReadNode(1, 0, 0, 4), \
    ReadNode(0, 7, 0, 8), ReadNode(1, 0, 1, 0), ReadNode(1, 0, 0, 10), \
    ReadNode(1, 0, 0, 11)]
    >>> generate_overall_tree_postorder(lst)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(None, \
HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(6, None, None)), HuffmanNode(4, None, None)), \
HuffmanNode(None, HuffmanNode(7, None, None), \
HuffmanNode(8, None, None))), HuffmanNode(10, None, None)), \
HuffmanNode(11, None, None))
    """
    # because we know that it is in postorder, we create a list, then
    # we take the first node, and if the left and right are both leaves,
    # we create a new huffmannode and append it to temp
    # if the right chid is an internal node, and left is a leaf, we createa node
    # by popping temp, and then add this node into temp.
    # if left is an internal node, and right is a leaf, do the opposite
    # if both left and right are internal nodes, create a node,
    # and pop the temp to add to right, then pop the temp to add to left

    temp = []
    for node in node_lst:
        if node.l_type == 0 and node.r_type == 0:
            temp.append(HuffmanNode(None, HuffmanNode(node.l_data),
                                    HuffmanNode(node.r_data)))
        elif node.r_type == 1 and node.l_type == 0:
            new_node = HuffmanNode(None, HuffmanNode(node.l_data), temp.pop())
            temp.append(new_node)
        elif node.l_type == 1 and node.r_type == 0:
            new_node = HuffmanNode(None, temp.pop(), HuffmanNode(node.r_data))
            temp.append(new_node)
        else:
            right = temp.pop()
            left = temp.pop()
            new_node = HuffmanNode(None, left, right)
            temp.append(new_node)

    return temp[0]


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes

    >>> t = HuffmanNode(None, HuffmanNode(None, HuffmanNode(3), \
    HuffmanNode(None, HuffmanNode(1), HuffmanNode(4))), \
    HuffmanNode(None, HuffmanNode(2), HuffmanNode(5)))
    >>> text = bytes([216, 0])
    >>> size = 4
    >>> generate_uncompressed(t, text, size) == bytes([5, 4, 3, 3])
    True
    >>> left1 = HuffmanNode(None, HuffmanNode(1), HuffmanNode(3))
    >>> left = HuffmanNode(None, left1, HuffmanNode(2))
    >>> right1 = HuffmanNode(None, HuffmanNode(10), HuffmanNode(11))
    >>> right = HuffmanNode(None, HuffmanNode(9), right1)
    >>> tree = HuffmanNode(None, left, right)
    >>> text = bytes([220, 184, 208, 64])
    >>> size = 10
    >>> generate_uncompressed(tree, text, size) == \
    bytes([10, 11, 3, 2, 10, 3, 9, 9, 1, 2])
    True
    """
    # make a string of all the bytes converted into bits in order to deal with
    # the nodes' codes' more easily
    bits = [byte_to_bits(byte) for byte in text]
    bit_string = ''
    for item in bits:
        bit_string += item

    # based on the numbers in bit_string, travel down the tree and appendd
    # the symbol once you get to a leaf
    i = 0
    symbols = []
    while len(symbols) < size:
        current = tree
        while current.left and current.right:
            if bit_string[i] == '0':
                current = current.left
                i += 1
            elif bit_string[i] == '1':
                current = current.right
                i += 1
        symbols.append(current.symbol)

    return bytes(symbols)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

# def improve_tree(tree, freq_dict):
#     """ Improve the tree as much as possible, without changing its shape,
#     by swapping nodes. The improvements are with respect to freq_dict.
#
#     @param HuffmanNode tree: Huffman tree rooted at 'tree'
#     @param dict(int,int) freq_dict: frequency dictionary
#     @rtype: NoneType
#
#     >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
#     >>> right = HuffmanNode(None, HuffmanNode(101), \
#     HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
#     >>> tree = HuffmanNode(None, left, right)
#     >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
#     >>> improve_tree(tree, freq)
#     >>> avg_length(tree, freq)
#     2.31
#     """
#     freq_list = []
#     for key in freq_dict:
#         freq_list.append([freq_dict[key], key])
#     freq_list.sort()
#
#     codes_list = []
#     codes = get_codes(tree)
#     for symbol in codes:
#         codes_list.append([codes[symbol], symbol])
#     codes_list.sort()
#     codes_list.reverse()
#
#     longest_list = []
#     longest = len(codes_list[0][0])
#     i = 0
#     while len(codes_list[i][0]) == longest:
#         longest_list.append(codes_list[i])
#         i += 1
#
#     least = []
#     j = len(longest_list)
#     for k in range(0, j):
#         least.append(freq_list[k])
#
#     lst3 = []
#     for item in longest_list:
#         lst3.append(HuffmanNode(item[1]))
#     # for item in longest_list:
#     #     current = tree
#     #     for num in item[0]:
#     #         if num == '0':
#     #             current = current.left
#     #         elif num == '1':
#     #             current = current.right
#     #     lst3.append(current)
#
#     for item in lst3:
#         node = item
#
#     def act(t):
#         """ Append t to lst if t.number is equal to root_index.
#
#         @param HuffmanNode t:
#         @rtype: None
#         """
#         if t == node:
#             t.symbol = least[0][1]
#             del least[0]
#
#     postorder_visit(tree, act)
#
#     print(freq_list)
#     print(codes_list)
#     print(longest_list)
#     print(least)
#     print(lst3)
#     print(tree)


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
