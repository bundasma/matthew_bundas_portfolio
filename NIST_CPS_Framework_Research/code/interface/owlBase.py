
'''
owlApplication

Defines a Class capturing the base ontology. It has as a data member
which is the actual owlready2 object, as well as supplemental attributes.

More extensive documentation is provided.

'''


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from owlNode import owlNode
from owlFunctions import remove_namespace
from owlready2 import *
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from owlFunctions import is_asp_or_conc

class owlBase:

    def __init__(self,filename):

        #owlready ontology member
        self.owlReadyOntology = None
        self.owlName = None

        #arrays for holding owlNodes
        self.nodeArray = None
        self.concernArray = None
        self.owlApplication = None
        self.allConcerns_owlNode = None
        self.aspectConcernArray = None

        #number of different types of objects
        self.numNodes = None
        self.numAspects = None
        self.numConcerns = None
        self.numComponents = None
        self.numConditions = None
        self.numImpactRules = None

        #graph associated with ontology
        self.owlGraph = None
        self.graphPositions = None

        #graph information
        self.subconcernEdges = None
        self.concernEdgeLabels = None
        self.aspectNodeLabels = None
        self.concernNodeLabels = None

        #dimensions of viewable area
        self.minX = None
        self.maxX = None
        self.minY = None
        self.maxY = None
        self.totalX = None
        self.totalY = None

        #load in passed filename
        self.initializeOwlOntology(filename)


    #loads ontology from owl file
    def initializeOwlOntology(self,filename):

        self.loadOwlFile(filename)

    #initialized owlNodes given owlready ontology
    def initializeOwlNodes(self):

        self.addOwlNodes()
        self.assignChildren()
        self.assignParentsFromChildren()

    #loads ontology file
    def loadOwlFile(self,filename):

        self.owlReadyOntology = get_ontology("file://./" + filename).load()
        self.owlName = str(filename)

    #updates number of different types of objects
    def setNumbers(self):

        self.numAspects =  len(self.owlReadyOntology.search(type = self.owlReadyOntology.Aspect))
        self.numConcerns =  len(self.owlReadyOntology.search(type = self.owlReadyOntology.Concern))
        self.numNodes = self.numAspects + self.numConcerns

    #finds the owlNode object of given name
    def findNode(self,name):

        for node in self.allConcerns_owlNode:

            if(node.name == name):

                return node

        print("couldn't find " + str(name))
        return 0

    #initializes owlNode objects from owlready ontology
    def addOwlNodes(self):

        self.allConcerns_owlNode = []

        #all instances of type Concern
        all_concerns = np.asarray(self.owlReadyOntology.search(type = self.owlReadyOntology.Concern))

        #create owlNode object for each concern, add it to lists
        for concern in all_concerns:

            newOwlNode = owlNode()
            newOwlNode.name = remove_namespace(concern)
            newOwlNode.type = remove_namespace(concern.is_a[0])
            newOwlNode.children = []
            newOwlNode.parents = []
            newOwlNode.owlreadyObj = concern

            self.allConcerns_owlNode.append(newOwlNode)

    #goes through all owlNodes, handles children data member
    def assignChildren(self):

        #loops through all concerns
        for node in self.allConcerns_owlNode:

            #gets subconcerns/children
            children_list = node.owlreadyObj.includesConcern

            #updates children data member
            for child in children_list:

                node.children.append(self.getOwlNode(remove_namespace(child)))

    #goes through all owlNodes, handles parents data member
    def assignParentsFromChildren(self):

        #loops through all concerns
        for node in self.allConcerns_owlNode:

            #updates parents data member
            for child in node.children:

                child.parents.append(node)

    #gets owlNode object given name
    def getOwlNode(self,name):

        #loops through all concerns, trying to find matching name
        for node in self.allConcerns_owlNode:

            if(node.name == name):

                return node

        #loops through application ontology if not found in base
        if(self.owlApplication != None):

            for node in self.owlApplication.nodeArray:

                if(node.name == name):

                    return node

        return 0

    #adds new concern to base ontology as subconcern
    def addNewConcern(self,new_name,clicked_name):

        #instantiate the object with the given new name
        new_concern = self.owlReadyOntology.Concern(new_name,ontology = self.owlReadyOntology)

        #add the new concern as a subconcern of the clicked node
        subconcern_of_owlNode = self.getOwlNode(clicked_name)

        #update includesConcern relation
        subconcern_of_owlreadyNode = subconcern_of_owlNode.owlreadyObj
        subconcern_of_owlreadyNode.includesConcern.append(new_concern)

    #adds new relationless concern
    def addNewRLConcern(self,new_name):

        new_concern = self.owlReadyOntology.Concern(new_name,ontology = self.owlReadyOntology)

    #adds a subconcern relation between two concerns
    def addNewSubConcernRelation(self,parent,child):

        if(is_asp_or_conc(parent.type) == False or is_asp_or_conc(child.type) == False):
            print("Either parent or child is not a concern")
            return


        parent.owlreadyObj.includesConcern.append(child.owlreadyObj)

    #adds concern as parent of another concern
    def addConcernAsParent(self,new_name,clicked_name):

        #instantiate new concern
        new_parent_concern = self.owlReadyOntology.Concern(new_name,ontology = self.owlReadyOntology)

        #get owlready object of selected node
        parent_of = self.getOwlNode(clicked_name)

        #sets up parent relation
        new_parent_concern.includesConcern.append(parent_of.owlreadyObj)

    #removes a subconcern relation between two concerns
    def removeSubConcernRelation(self,parent,child):

         if(is_asp_or_conc(child.type) == False or is_asp_or_conc(parent.type) == False):
            print("One node is not a concern")
            return

         print("removing relation between " + parent.name + " and " + child.name)

         parent.owlreadyObj.includesConcern.remove(child.owlreadyObj)

    #adds relationless aspect
    def addNewAspect(self,new_name):

        new_aspect = self.owlReadyOntology.Aspect(new_name,ontology = self.owlReadyOntology)


    def editName(self,current_obj,new_name):

        current_obj.owlreadyObj.name = new_name

    def removeConcern(self,to_remove):

        destroy_entity(to_remove.owlreadyObj)

    #removes all relationless children
    def removeRelationless(self):

        for node in self.allConcerns_owlNode:

            #node is relationlesss if it has no parents or children
            if(len(node.parents) == 0 and len(node.children) == 0 and node.type != "Aspect"):

                print("trying to remove " + node.name)

                #get owlready objects to remove
                to_remove = self.owlReadyOntology.ontology.search(iri = "*" + node.name)

                names = []

                #get names without namespace
                for subc in to_remove:
                    names.append(remove_namespace(subc))

                i = 0

                #find matching names
                while i < len(names):

                    if(names[i] == node.name):
                        break

                    i = i + 1

                #remove owlready object
                to_remove = to_remove[i]
                destroy_entity(to_remove)

    #finds all relationless recursively, adds them to list
    def findRelationless(self,parent,relationless):

        #stop if no children to continue with
        if(len(parent.children) == 0):
            return

        #recursively call again on children
        for child in parent.children:

            if(len(child.parents) == 1 and child.parents[0] == parent ):

                relationless.append(child)

            self.findRelationless(child,relationless)

    #recursively finds all children, adds them to list
    def findChildren(self,parent,all_children):

        #stop if no parents to continue with
        if(len(parent.children) == 0):
            return

        #recursively call on parents
        for child in parent.children:

            all_children.append(child)

            self.findChildren(child,all_children)


    #front-facing function to find all relationless
    def getRelationless(self,parent):

        relationless = []


        self.findRelationless(parent,relationless)

        return relationless

    #front-facing function to find all children of a parent
    def getChildren(self,parent):

        all_children = []

        self.findChildren(parent,all_children)

        return all_children
