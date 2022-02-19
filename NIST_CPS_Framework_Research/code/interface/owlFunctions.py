'''
owlFunctions

Defines supplementary functions for general purpose

More extensive documentation is provided.

'''

from parse import *
import networkx as nx


#checks whether the passed string matches Concern or Aspect
def is_asp_or_conc(mytype):

    if(not (mytype == "Concern" or mytype == "Aspect") ):
        return False
    else:
        return True

#removes the namespace in front of strings produced from owlready objects
#ie namespace.concernName -> concernName
def remove_namespace(in_netx):

    #convert owlready object to string
    in_str = str(in_netx)

    #find location of period
    leng = len(in_str)
    period = leng
    for i in range(leng):
        if(in_str[i] == '.'):
            period = i
            break

    #returns all characters after period
    return in_str[(period + 1):]




#from https://stackoverflow.com/questions/15353087/programmatically-specifying-nodes-of-the-same-rank-within-networkxs-wrapper-for
def graphviz_layout_with_rank(G, prog = "neato", root = None, sameRank = [], args = ""):
    ## See original import of pygraphviz in try-except block
    try:
        import pygraphviz
    except ImportError:
        raise ImportError('requires pygraphviz ',
                          'http://pygraphviz.github.io/')
    ## See original identification of root through command line

    if root is not None:
        args += f"-Groot={root}"


    A = nx.nx_agraph.to_agraph(G)
    for sameNodeHeight in sameRank:
        if type(sameNodeHeight) == str:
            print("node \"%s\" has no peers in its rank group" %sameNodeHeight)
        A.add_subgraph(sameNodeHeight, rank="same")
    A.layout(prog=prog, args=args)
    ## See original saving of each node location to node_pos

    node_pos = {}
    for n in G:
        node = pygraphviz.Node(A, n)
        try:
            xs = node.attr["pos"].split(',')
            node_pos[n] = tuple(float(x) for x in xs)
        except:
            print("no position for node", n)
    return node_pos
