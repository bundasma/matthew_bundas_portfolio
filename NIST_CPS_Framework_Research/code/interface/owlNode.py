'''
owlNode

Defines a Class capturing an indvidual in an ontology. Is the basic building block
of the editor. Each individual in both the base and application ontologies gets an
instance of owlNode. Its attributes describe the characteristics of the indvidual,
including its name, type, parents, children, position, and has a pointer to
its owlready object

More extensive documentation is provided.

'''


class owlNode:

    def __init__(self):

       self.name = None #name of individual
       self.type = None #type of individual, concern, property etc
       self.owlreadyObject = None #pointer to owlready instance
       self.parent = None #list of parent nodes
       self.children = None #list of children nodes
       self.negChildren = None #list of negated children
       self.relevantProperties = None #list of properties connected to individual

       self.subtype = "None" #subtype if applicable, ie Formula can be conjunction or disjunction

       self.xpos = None #x position in networkx graph, helpful for hovering/click handling
       self.ypos = None #y position in networkx graph, helpful for hovering/click handling
       self.level = None #depth in networkx graph

       self.polarity = "neutral" #default to neutral polarity, this is mostly outdated
