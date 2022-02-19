'''
owlGraph

Defines a Class capturing the graph displayed in the OntologyGUI. It has members
pointing to the owlBase and owlApplication and arrays holding the owlNodes in these
classes, as well as arrays holding the edges between these nodes. It works
with NetworkX and Matplotlib to draw nodes on a graph, and draw the edges between
them.

More extensive documentation is provided.

'''

from owlBase import owlBase
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from owlNode import owlNode
from script_networkx import remove_namespace
from owlready2 import *
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from owlFunctions import graphviz_layout_with_rank


class owlGraph:

    def __init__(self,baseOWL,appOWL = None):

        #flags to control whether we want to graph properties/components
        self.graphProperties = True
        self.graphComponents = True

        #owlBase and owlApplications we are graphing
        self.owlBase = baseOWL
        self.owlApplication = appOWL

        #positions of nodes
        self.graphPositions = None

        #arrays of nodes
        self.aspectConcernArray = None
        self.aspectNameList = None
        self.formulasArray = None
        self.decompFuncArray = None
        self.componentArray = None

        #dictionaries of edges
        self.subconcernEdges = None
        self.concernFormulasEdges = None
        self.formulasDecompFuncEdges = None
        self.formulasPropertyEdges = None
        self.componentEdges = None

        #lists of edge labels
        self.concernEdgeLabels = None
        self.concernFormulasEdgeLabels = None
        self.formulasDecompFuncEdgeLabels = None
        self.formulasPropertyEdgeLabels = None
        self.componentEdgeLabels = None

        #lists of node labels
        self.aspectNodeLabels = None
        self.concernNodeLabels = None
        self.propertyNodeLabels = None
        self.formulasNodeLabels = None
        self.decompFuncNodeLabels = None
        self.componentNodeLabels = None

        #coordinates of graph dimensions
        self.minX = None
        self.maxX = None
        self.minY = None
        self.maxY = None
        self.totalX = None
        self.totalY = None
        self.XYRatio = None

        #construct graph given passed base/application
        self.makeGraph()

    #handles the construction of a graph
    def makeGraph(self):

        self.netXGraph = nx.DiGraph()

        self.addGraphNodes()
        self.addGraphEdgesAndLabels()
        self.addNodeLabels()
        self.setPositions()

    #loops through all nodes in base, application, adds them to graph
    def addGraphNodes(self):

        self.aspectConcernArray = np.array(())
        self.aspectNameList = [[]]
        self.propertyArray = np.array(())
        self.formulasArray = np.array(())
        self.decompFuncArray = np.array(())
        self.componentArray = np.array(())

        #loop through base ontology
        for node in self.owlBase.allConcerns_owlNode:

            #add node to graph
            self.netXGraph.add_node(node.name)

            #add node to array to keep track
            if(str(node.type) == "Concern" or str(node.type) == "Aspect"):

                self.aspectConcernArray = np.append(self.aspectConcernArray,node)

                #keep track of aspects for special graphing
                if(str(node.type) == "Aspect"):
                    self.aspectNameList[0].append(node.name)

            else:
                print("couldnt find type")
                print(node.type)

        #if we have an owlApplication, add those nodes too
        if(self.owlApplication != None):

            #loop through all nodes, handle them depending on type, adding each
            #to the graph, and array
            for node in self.owlApplication.nodeArray:


                if(str(node.type) == "Property" and self.graphProperties == True):

                    self.netXGraph.add_node(node.name)
                    self.propertyArray = np.append(self.propertyArray,node)

                elif(str(node.type) == "Formulas" and self.graphProperties == True):
                    #print(node.name)
                    #print("found formulas " + node.name)
                    self.netXGraph.add_node(node.name)
                    self.formulasArray = np.append(self.formulasArray,node)
                    continue

                elif(str(node.type) == "DecompositionFunction" and self.graphProperties == True):
                    self.netXGraph.add_node(node.name)
                    self.decompFuncArray = np.append(self.decompFuncArray,node)

                elif(str(node.type) == "Component" and self.graphComponents == True):
                    self.netXGraph.add_node(node.name)
                    self.componentArray = np.append(self.componentArray,node)

                else:

                    print("not graphing ", node.name," - ", node.type)
                    continue


    #handles adding of edges and labels to networkx graph
    def addGraphEdgesAndLabels(self):

        #each type of edge gets its own list

        self.subconcernEdges = []
        self.concernFormulasEdges = []

        self.formulasDecompFuncDisEdges = []
        self.formulasDecompFuncConjEdges = []

        self.formulasPropertyDisEdges = []
        self.formulasPropertyConjEdges = []

        self.negFormulasPropertyDisEdges = []
        self.negFormulasPropertyConjEdges = []

        self.concernPropertyEdges = []
        self.componentEdges = []

        self.concernEdgeLabels = {}
        self.concernFormulasEdgeLabels = {}

        self.formulasDecompFuncDisEdgeLabels = {}
        self.formulasDecompFuncConjEdgeLabels = {}

        self.formulasPropertyDisEdgeLabels = {}
        self.formulasPropertyConjEdgeLabels = {}

        self.negFormulasPropertyDisEdgeLabels = {}
        self.negFormulasPropertyConjEdgeLabels = {}

        self.concernPropertyEdgeLabels = {}
        self.componentEdgeLabels = {}

        #loop through each concern node, adding edges for its children
        for node in self.owlBase.allConcerns_owlNode:

            #if no children, move on
            if len(node.children) == 0:
                continue

            #loop through each child, adding edge depending on types
            for child in node.children:

                #subconcern edge
                if(child.type == "Concern"):

                    self.netXGraph.add_edge(node.name,child.name,length = 1)
                    self.subconcernEdges.append((node.name,child.name))
                    self.concernEdgeLabels[(node.name,child.name)] = 'subconcern'

                #formula addressing concern edges
                elif(child.type == "Formulas" or child.type == "DecompositionFunction"):

                    self.netXGraph.add_edge(node.name,child.name,length = 1)
                    self.concernFormulasEdges.append((node.name,child.name))
                    self.concernFormulasEdgeLabels[(node.name,child.name)] = "addressesConcern"

                #property addressing concern draw_networkx_edges
                elif(child.type == "Property" and len(child.parents) == 1):

                    self.netXGraph.add_edge(node.name,child.name, length = 1)
                    self.concernPropertyEdges.append((node.name,child.name))
                    self.concernPropertyEdgeLabels[(node.name,child.name)] = "addressesConcern"

        #do nothing if theres no application ontology
        if(self.owlApplication == None):
             return

        #loop through all nodes in application ontology
        for node in self.owlApplication.nodeArray:

            #if no children, move on
            if (len(node.children) + len(node.negChildren)) == 0:
                continue

            #handle adding edges with negated children
            for negChild in node.negChildren:

                if((negChild.type == "DecompositionFunction" or negChild.type == "Formulas") and self.graphProperties == True):

                     self.netXGraph.add_edge(node.name,negChild.name,length = 1)

                     #keep track of which are disjunctions and conjunctions
                     if(node.subtype == "Disjunction"):

                        self.negFormulasPropertyDisEdges.append((node.name,negChild.name))
                        self.negFormulasPropertyDisEdgeLabels[(node.name,negChild.name)] = "negMemberOf"

                     else:

                        self.negFormulasPropertyConjEdges.append((node.name,negChild.name))
                        self.negFormulasPropertyConjEdgeLabels[(node.name,negChild.name)] = "negMemberOf"


                #add edges between properties
                if(negChild.type == "Property" and self.graphProperties == True):

                    self.netXGraph.add_edge(node.name,negChild.name,length = 1)

                    #keep track of which are disjunctions and conjunctions to graph them differently
                    if(node.subtype == "Disjunction"):

                        self.negFormulasPropertyDisEdges.append((node.name,negChild.name))
                        self.negFormulasPropertyDisEdgeLabels[(node.name,negChild.name)] = "negMemberOf"

                    else:

                        self.negFormulasPropertyConjEdges.append((node.name,negChild.name))
                        self.negFormulasPropertyConjEdgeLabels[(node.name,negChild.name)] = "negMemberOf"

            #loop through all children, add their edges
            for child in node.children:

                if((child.type == "DecompositionFunction" or child.type == "Formulas") and self.graphProperties == True):

                    self.netXGraph.add_edge(node.name,child.name,length = 1)

                    #keep track of disjunction and conjunction to graph them differently
                    if(node.subtype == "Disjunction"):

                        self.formulasDecompFuncDisEdges.append((node.name,child.name))
                        self.formulasDecompFuncDisEdgeLabels[(node.name,child.name)] = "memberOf"

                    else:

                        self.formulasDecompFuncConjEdges.append((node.name,child.name))
                        self.formulasDecompFuncConjEdgeLabels[(node.name,child.name)] = "memberOf"

                #add edges between properties
                elif(child.type == "Property" and self.graphProperties == True):

                    self.netXGraph.add_edge(node.name,child.name,length = 1)

                    #keep track of disjunction and conjunction to graph them differently
                    if(node.subtype == "Disjunction"):

                        self.formulasPropertyDisEdges.append((node.name,child.name))
                        self.formulasPropertyDisEdgeLabels[(node.name,child.name)] = "memberOf"

                    else:

                        self.formulasPropertyConjEdges.append((node.name,child.name))
                        self.formulasPropertyConjEdgeLabels[(node.name,child.name)] = "memberOf"

                #add edges between components
                elif(child.type == "Component" and self.graphComponents == True):

                        self.netXGraph.add_edge(node.name,child.name,length = 1)

                        self.componentEdges.append((node.name,child.name))
                        self.componentEdgeLabels[(node.name,child.name)] = "relatedTo"
                else:
                    print(node.name)

    #handles adding of node labels to graph
    def addNodeLabels(self):

        #dictionaries for node labels, where both the key and value are the node name
        self.aspectNodeLabels = {}
        self.concernNodeLabels = {}
        self.propertyNodeLabels = {}
        self.formulasNodeLabels = {}
        self.decompFuncNodeLabels = {}
        self.componentNodeLabels = {}

        #loop through base nodes, add concern/aspect labels
        for node in self.aspectConcernArray:

            #aspects graphed differently than regular concerns
            if(node.type == "Aspect"):
                self.aspectNodeLabels[node.name] = node.name
            if(node.type == "Concern"):
                self.concernNodeLabels[node.name] = node.name

        #stop if there's no application ontology
        if(self.owlApplication == None):
             return

        #loop through application ontology
        for node in self.owlApplication.nodeArray:

                if(node.type == "Component" and self.graphComponents == True):

                    self.componentNodeLabels[node.name] = node.name

                elif(node.type == "Property" and self.graphProperties == True):

                    self.propertyNodeLabels[node.name] = node.name

                elif(node.type == "Formulas" and self.graphProperties == True):

                    self.formulasNodeLabels[node.name] = node.name

                elif(node.type == "DecompositionFunction" and self.graphProperties == True):

                    self.decompFuncNodeLabels[node.name] = node.name

    #handles setting of node positions in graph, using built in functions from graphviz
    def setPositions(self):

        #have graphviz create node positions in tree layout, with aspects at top "sameRank"
        self.graphPositions = graphviz_layout_with_rank(self.netXGraph, prog='dot',sameRank=self.aspectNameList)

        #don't want negative position values, so add 500 to all x's, 100 to all y's
        for x in self.graphPositions:

            lst = list(self.graphPositions[x])
            lst[0] = lst[0] + 500
            lst[1] = lst[1] + 100

            self.graphPositions[x] = tuple(lst)

        #find x,y mins and maxes for gui graphing purposes
        x_pos = np.array(())
        y_pos = np.array(())

        #set the actual owlNode positions, used for hovering/click detection in GUI
        for x in self.graphPositions:

            #position tuple of a node
            position = self.graphPositions[x]

            #extract x/y positions
            x_pos = np.append(x_pos,position[0])
            y_pos = np.append(y_pos,position[1])

            #find owlNode
            mynode = self.findNode(x)

            #update owlNode's position
            mynode.xpos = position[0]
            mynode.ypos = position[1]


        #find dimensions of graph window
        xmax = np.max(x_pos)
        ymax = np.max(y_pos)

        xmin = np.min(x_pos)
        ymin = np.min(y_pos)

        totalx_o = xmax - xmin
        totaly_o = ymax - ymin

        self.minX = xmin - totalx_o/10
        self.maxX = xmax + totalx_o/10

        self.minY = ymin - totaly_o/10
        self.maxY = ymax + totaly_o/10

        self.totalX = self.maxX - self.minX
        self.totalY = self.maxY - self.minY

        self.XYRatio = self.totalX/self.totalY



    #handles the actual graphing of the tree after we have added all of the nodes/edges/labels
    def draw_graph(self, ax,fs):

        aspect_color = "#000a7d" #blue
        concern_color = "#800000" #red

        property_color = "#595858" #gray
        formulas_color = "#3000ab" #purple
        decompfunc_color ="#3000ab" #purple

        component_color = "pink"
        edge_color = "black"
        edge_width = 2
        edge_alpha = .8

        #scale font size
        fs = fs*.69

        plt.tight_layout()

        #draw nodes
        nx.draw_networkx_nodes(self.netXGraph, pos = self.graphPositions, with_labels=False, arrows=False, ax = ax, node_size = .1,scale = 1)

        #draw edges
        nx.draw_networkx_edges(self.netXGraph, pos = self.graphPositions, edgelist = self.subconcernEdges, arrows=False,style = "solid",width = edge_width,edge_color = edge_color,alpha = edge_alpha)
        nx.draw_networkx_edges(self.netXGraph, pos = self.graphPositions, edgelist = self.concernFormulasEdges, arrows=False,style = "solid",width = edge_width,edge_color = edge_color, alpha = edge_alpha)
        nx.draw_networkx_edges(self.netXGraph, pos = self.graphPositions, edgelist = self.concernPropertyEdges, arrows=False,style = "solid",width = edge_width,edge_color = edge_color,alpha = edge_alpha)
        nx.draw_networkx_edges(self.netXGraph, pos = self.graphPositions, edgelist = self.formulasDecompFuncDisEdges, arrows=False,style = "dotted",width = edge_width,edge_color = edge_color, alpha = edge_alpha)
        nx.draw_networkx_edges(self.netXGraph, pos = self.graphPositions, edgelist = self.formulasDecompFuncConjEdges, arrows=False,style = "solid",width = edge_width,edge_color = edge_color, alpha = edge_alpha)
        nx.draw_networkx_edges(self.netXGraph, pos = self.graphPositions, edgelist = self.formulasPropertyDisEdges, arrows=False,style = "dotted",width = edge_width,edge_color = edge_color, alpha = edge_alpha)
        nx.draw_networkx_edges(self.netXGraph, pos = self.graphPositions, edgelist = self.formulasPropertyConjEdges, arrows=False,style = "solid",width = edge_width,edge_color = edge_color, alpha = edge_alpha)
        nx.draw_networkx_edges(self.netXGraph, pos = self.graphPositions, edgelist = self.negFormulasPropertyDisEdges, arrows=False,style = "dotted",width = edge_width,edge_color = "red", alpha = edge_alpha)
        nx.draw_networkx_edges(self.netXGraph, pos = self.graphPositions, edgelist = self.negFormulasPropertyConjEdges, arrows=False,style = "solid",width = edge_width,edge_color = "red", alpha = edge_alpha)
        nx.draw_networkx_edges(self.netXGraph, pos = self.graphPositions, edgelist = self.componentEdges, arrows=False,style = "dotted",width = edge_width,edge_color = edge_color, alpha = edge_alpha)

        #draw edge labels
        nx.draw_networkx_edge_labels(self.netXGraph, pos = self.graphPositions, edge_labels=self.concernEdgeLabels,font_size = fs)
        nx.draw_networkx_edge_labels(self.netXGraph, pos = self.graphPositions, edge_labels=self.concernFormulasEdgeLabels,font_size = fs)
        nx.draw_networkx_edge_labels(self.netXGraph, pos = self.graphPositions, edge_labels=self.concernPropertyEdgeLabels,font_size = fs)
        nx.draw_networkx_edge_labels(self.netXGraph, pos = self.graphPositions, edge_labels=self.formulasDecompFuncDisEdgeLabels,font_size = fs)
        nx.draw_networkx_edge_labels(self.netXGraph, pos = self.graphPositions, edge_labels=self.formulasDecompFuncConjEdgeLabels,font_size = fs)
        nx.draw_networkx_edge_labels(self.netXGraph, pos = self.graphPositions, edge_labels=self.formulasPropertyDisEdgeLabels,font_size = fs)
        nx.draw_networkx_edge_labels(self.netXGraph, pos = self.graphPositions, edge_labels=self.formulasPropertyConjEdgeLabels,font_size = fs)
        nx.draw_networkx_edge_labels(self.netXGraph, pos = self.graphPositions, edge_labels=self.negFormulasPropertyDisEdgeLabels,font_size = fs)
        nx.draw_networkx_edge_labels(self.netXGraph, pos = self.graphPositions, edge_labels=self.negFormulasPropertyConjEdgeLabels,font_size = fs)
        nx.draw_networkx_edge_labels(self.netXGraph, pos = self.graphPositions, edge_labels=self.componentEdgeLabels,font_size = fs)

        #draw base node labels
        nx.draw_networkx_labels(self.netXGraph,self.graphPositions,self.aspectNodeLabels,font_size=fs,bbox=dict(facecolor=aspect_color, boxstyle='square,pad=.3'),font_color = "white")
        nx.draw_networkx_labels(self.netXGraph,self.graphPositions,self.concernNodeLabels,font_size= fs,bbox=dict(facecolor=concern_color, boxstyle='square,pad=.3'),font_color = "white")

        #draw application node labels
        if(self.owlApplication != None):

            nx.draw_networkx_labels(self.netXGraph,self.graphPositions,self.propertyNodeLabels,font_size=fs*.90,bbox=dict(facecolor=property_color, boxstyle='round4,pad=.3'),font_color = "white")

            #draw workshop ontologies with smaller fontsize
            list_names = ["workshop_ontologies/cpsframework-v3-sr-LKAS-Configuration-V1.owl","workshop_ontologies/cpsframework-v3-blank-app.owl"]
            if(self.owlApplication.owlName in list_names):

                nx.draw_networkx_labels(self.netXGraph,self.graphPositions,self.formulasNodeLabels,font_size=fs*.90,bbox=dict(facecolor=property_color, boxstyle='round4,pad=.3'),font_color = "white")

            else:
                nx.draw_networkx_labels(self.netXGraph,self.graphPositions,self.formulasNodeLabels,font_size=fs,bbox=dict(facecolor=formulas_color, boxstyle='round4,pad=.3'),font_color = "white")

            nx.draw_networkx_labels(self.netXGraph,self.graphPositions,self.decompFuncNodeLabels,font_size=fs,bbox=dict(facecolor=decompfunc_color, boxstyle='round4,pad=.3'),font_color = "white")
            nx.draw_networkx_labels(self.netXGraph,self.graphPositions,self.componentNodeLabels,font_size=fs,bbox=dict(facecolor=component_color, boxstyle='round,pad=.3'),font_color = "white")

    #finds an owlNode with the given name
    def findNode(self,name):

        #look in base ontology
        for node in self.owlBase.allConcerns_owlNode:

            if(node.name == name):
                return node

        #look in application ontology
        for node in self.owlApplication.nodeArray:

            if(node.name == name):
                return node


        print("couldn't find " + str(name))
        return 0
