
'''
owlApplication

Defines a Class capturing the application ontology. It has as a data member
which is the actual owlready2 object, as well as supplemental attributes.

More extensive documentation is provided.

'''

from owlready2 import *
import numpy as np
from owlNode import owlNode
from owlFunctions import remove_namespace
from owlFunctions import remove_ir
from owlBase import owlBase

class owlApplication:

    def __init__(self,filename,base):

        #support properties and components by default
        self.handleProperties = True
        self.handleComponents = True

        #no application ontology yet
        self.owlreadyOntology = None

        #the corresponding owlBase instance
        self.owlBase = base

        #connect base with application
        self.owlBase.owlApplication = self

        #initialize data members
        self.owlName = None

        #list of nodes
        self.nodeArray = None
        self.allComponents_owlNode = None
        self.allProperties_owlNode = None
        self.allFormulas_owlNode = None
        self.allDecompFuncs_owlNode = None

        self.numNodes = None
        self.numComponents = None
        self.numProperties = None

        #graph application belongs to
        self.owlGraph = None

        #load from the given .owl file
        self.loadOwlFile(filename)

    #has owlready2 load from .owl file, start owlready ontology
    def loadOwlFile(self,filename):

        #loads ontology from file
        self.owlreadyOntology = get_ontology("file://./" + filename).load()

        #sets name of ontology
        self.owlName = str(filename)
        print(self.owlName)


    #intializes owlNodes from the owlready2 ontology
    def initializeOwlNodes(self):

        self.nodeArray = []

        #add components
        if(self.handleComponents == True):
            self.addComponents()

        #add properties/formulas
        if(self.handleProperties == True):

            self.addProperties()
            self.addFormulas()
            self.addFuncDecomps()

        #add components
        if(self.handleComponents == True):

            self.handleRelateToProperty()

        #try handling relations if they exist
        if(self.handleProperties == True):

            try:
                self.handleMemberOf()

            except:
                print("couldn't ao memberof")

            try:
                self.handleNegMemberOf()

            except:
                print("couldn't do negmemberof")

            try:
                self.handleFormulasAddConcern()

            except:
                print("couldn't do handleformulasaddconcern")

            self.handleaddConcern()


    #adds property owlNodes to list from owlready ontology
    def addProperties(self):

        self.allProperties_owlNode = []

        #get all owlready objects of type property
        all_props = np.asarray(self.owlreadyOntology.search(type = self.owlreadyOntology.Property))

        #initialize an owlNode for each property, add it to a list
        for prop in all_props:


            newOwlNode = owlNode()
            newOwlNode.name = remove_namespace(prop)
            newOwlNode.type = "Property"
            newOwlNode.children = []
            newOwlNode.negChildren = []
            newOwlNode.parents = []

            newOwlNode.owlreadyObj = prop

            self.allProperties_owlNode.append(newOwlNode)
            self.nodeArray.append(newOwlNode)

    #adds formula owlNodes to list from owlready ontology
    def addFormulas(self):

        self.allFormulas_owlNode = []

        #get all owlready objects of type property/formulas/decomps
        props = np.asarray(self.owlreadyOntology.search(type =  self.owlreadyOntology.Property))
        forms = np.asarray(self.owlreadyOntology.search(type =  self.owlreadyOntology.Formulas))
        decomps = np.asarray(self.owlreadyOntology.search(type = self.owlreadyOntology.DecompositionFunction))

        #list of all properties/formulas
        all_Formulas = list(set(props) ^ set(forms) ^ set(decomps))

        #initialize an owlNode for each individual, add it to a list
        for Formulas in all_Formulas:

            #ignore formulas that start with g, they are of outdated form
            if(remove_namespace(Formulas)[0] == "g"):

                continue


            newOwlNode = owlNode()
            newOwlNode.name = remove_namespace(Formulas)
            newOwlNode.type = "Formulas"
            newOwlNode.subtype = self.getSubType(Formulas)
            newOwlNode.children = []
            newOwlNode.negChildren = []
            newOwlNode.parents = []
            newOwlNode.owlreadyObj = Formulas

            self.allFormulas_owlNode.append(newOwlNode)
            self.nodeArray.append(newOwlNode)


    #adds component owlNodes to list from owlready ontology
    def addComponents(self):

        self.allComponents_owlNode = []

        #get all owlready objects of type component
        all_components = np.asarray(self.owlreadyOntology.search(type = self.owlreadyOntology.Component))

        #initialize an owlNode for each individual, add it to a list
        for component in all_components:

            newOwlNode = owlNode()
            newOwlNode.name = remove_namespace(component)
            newOwlNode.type = remove_namespace(component.is_a[0])
            newOwlNode.children = []
            newOwlNode.negChildren = []
            newOwlNode.parents = []
            newOwlNode.owlreadyObj = component

            self.allComponents_owlNode.append(newOwlNode)
            self.nodeArray.append(newOwlNode)


    #handles memberOf relation, sets parent/child data members of owlNodes
    def handleMemberOf(self):

        #for each node in nodeArray, find its parents, set children/parents attributes
        for child in self.nodeArray :

            #get owlNodes owlready object
            child_owlr = child.owlreadyObj

            #get owlready object's memberOf owlready object
            member_of = child_owlr.memberOf

            #for reach memberOf object (parent), add set children/parent attributes
            for memberof in member_of:

                #get parent owlNode object
                parent = self.getOwlNode(remove_namespace(memberof))

                #set child's parent
                child.parents.append(parent)

                #set parent's child
                parent.children.append(child)


    #handles negMemberOf relation, sets parent/child data members of owlNodes
    def handleNegMemberOf(self):

        #for each node in nodeArray, find its parents, set children/parents attributes
        for child in self.nodeArray:

            #get owlNodes owlready object
            child_owlr = child.owlreadyObj

            #get owlready object's negMemberOf owlready object
            neg_member_of = child_owlr.negMemberOf

            #for reach negMemberOf object (parent), add set children/parent attributes
            for negmemberof in neg_member_of:

                #get parent owlNode object
                parent = self.getOwlNode(remove_namespace(negmemberof))

                #set child's parent
                child.parents.append(parent)

                #set parent's child
                parent.negChildren.append(child)


    #handles formulasAddConcern relation, sets parent/child data members of owlNodes
    def handleFormulasAddConcern(self):

        #for each node in nodeArray, find its parents, set children/parents attributes
        for child in self.nodeArray:

            #get owlNodes owlready object
            child_owlr = child.owlreadyObj

            #get owlready object's formulasAddConcern owlready object
            addresses = child_owlr.formulasAddConcern

            #for reach addressed object (parent), add set children/parent attributes
            for addressed in addresses:

                #get parent owlNode object
                parent = self.getOwlNodeFromBase(remove_namespace(addressed))

                #set child's parent
                child.parents.append(parent)

                #set parent's child
                parent.children.append(child)

    #handles formulasAddConcern relation, sets parent/child data members of owlNodes
    def handleaddConcern(self):

        #for each node in nodeArray, find its parents, set children/parents attributes
        for child in self.nodeArray:

            #get owlNodes owlready object
            child_owlr = child.owlreadyObj

            #get owlready object's addConcern owlready object
            addresses = child_owlr.addConcern

            #for reach addressed object (parent), add set children/parent attributes
            for addressed in addresses:

                #get parent owlNode object
                parent = self.getOwlNodeFromBase(remove_namespace(addressed))

                #set child's parent
                child.parents.append(parent)
                #set parent's child
                parent.children.append(child)

    #handles relateToProperty relation, sets parent/child data members of owlNodes
    def handleRelateToProperty(self):

        #for each component, find its parents, set children/parents attributes
        for comp in self.allComponents_owlNode:

            #get owlready object's addConcern owlready object
            all_rel_props = comp.owlreadyObj.relateToProperty

            #prop is owlready node
            for prop in all_rel_props:

                #get parent owlNode object
                prop_owlNode = self.getOwlNode(remove_namespace(prop))

                #set child's parent
                prop_owlNode.children.append(comp)
                #set parent's child
                comp.parents.append(prop_owlNode)


    #handles adding of property as the child of a concern
    def addPropertyAsChildofConcern(self,parentConcern,new_property_name):

        #new property owlready object
        new_property = self.owlreadyOntology.Property(new_property_name, ontology = self.owlreadyOntology)

        #properties are also formulas
        new_property.is_a.append(self.owlreadyOntology.Formulas)

        #add addresses concern relation
        new_property.addConcern.append(parentConcern.owlreadyObj)

    #handles adding of property as the parent of a concern
    def addPropertyAsParentofComponent(self,new_name,parent):

        #parent is component

        #new property owlready object
        new_property =  self.owlreadyOntology.Property(new_name, ontology = self.owlreadyOntology)

        #properties are also formulas
        new_property.is_a.append(self.owlreadyOntology.Formulas)

        #add addresses concern relation
        parent.owlreadyObj.relateToProperty.append(new_property)

    #handles adding relationless property
    def addRLProperty(self,new_name):

        #new propert owlready object
        new_property = self.owlreadyOntology.Property(new_name,ontology = self.owlreadyOntology)

        #properties are also formulas
        new_property.is_a.append(self.owlreadyOntology.Formulas)

    #handles adding new component
    def addNewComponent(self,new_name):

        #new component owlready object
        new_component = self.owlreadyOntology.Component(new_name,ontology = self.owlreadyOntology)

    #handles adding new component as child of a property
    def addNewComponentAsChild(self,new_name,parent):

        #new component owlready object
        new_component = self.owlreadyOntology.Component(new_name,ontology = self.owlreadyOntology)

        #adds relateToProperty relation
        new_component.relateToProperty.append(parent.owlreadyObj)

    #handles adding addConcernRelation
    def addaddConcernRelation(self,parent,child):

        #adds relation between parent and child
        child.owlreadyObj.addConcern.append(parent.owlreadyObj)

    #handles adding formulasAddConcern relation
    def addNewConcernFormulasRelation(self,parent,child):

        #parent is concern
        #child is Formulas

        #adds relation between parent and child
        child.owlreadyObj.formulasAddConcern.append(parent.owlreadyObj)

    #handles adding formulasAddConcern relation
    def addFormulasPropertyRelations(self,parent,child):

        #parent is Formulas
        #child is property

        #adds memberOf relation to child
        child.owlreadyObj.memberOf.append(parent.owlreadyObj)

        #adds includeMember relation to parent
        parent.owlreadyObj.includesMember.append(child.owlreadyObj)

        #gets concerns the parent addresses
        self.findAddressedConcern(parent.owlreadyObj)

        #adds addConcern relation to property
        for addconcern in self.addressed_concerns:

            child.owlreadyObj.addConcern.append(addconcern)

    #handles adding addFormulasFormulas relation
    def addFormulasFormulasRelations(self,parent,child):

        #parent is Formulas/functionaldecomp
        #child is Formulas/functionaldecomp

        child.owlreadyObj.memberOf.append(parent.owlreadyObj)
        parent.owlreadyObj.includesMember.append(child.owlreadyObj)

    #handles adding a new property to a formula
    def addNewPropertyToFormulas(self,parent,child_name):

        #creates new property
        new_property = self.owlreadyOntology.Property(child_name, ontology = self.owlreadyOntology)

        #properties are also formulas
        new_property.is_a.append(self.owlreadyOntology.Formulas)

        #formula is the parent of the property
        Formulas_owlready = parent.owlreadyObj

        #property is a member of the formula
        new_property.memberOf.append(Formulas_owlready)

        #formula includes the property
        Formulas_owlready.includesMember.append(new_property)

        #finds concerns the formula addresses
        self.findAddressedConcern(Formulas_owlready)

        #new property addresses all concerns the formula does (indirectly)
        for addconcern in self.addressed_concerns:

            new_property.addConcern.append(addconcern)

    #handles addding new relateToProperty relation
    def addNewRelatedToRelation(self,parent,child):

        child.owlreadyObj.relateToProperty.append(parent.owlreadyObj)

    #handles adding new dependency
    def addNewDependency(self,LHSNodes,RHSNode):

        #the most recent formula added
        last_Formulas = None

        #which number formula we are on, to create unique default names
        formulas_number = 1

        #for each node in the LHS, add a conjunction if its connected by and, dijunction if connected by or
        for lhsnode in LHSNodes:

            formulas_number +=1

            #add conjunction if operator is and
            if(lhsnode.operator == "and"):

                new_Formulas = self.owlreadyOntology.Conjunction(lhsnode.name,ontology = self.owlreadyOntology)

            #add disjunction if operation is not and
            else:

                new_Formulas = self.owlreadyOntology.Disjunction(lhsnode.name,ontology = self.owlreadyOntology)

            #update most recent formula
            last_Formulas = new_Formulas

            #for each member of the node, update relations
            for member in lhsnode.members:

                #if not a member of anything, do nothing
                if(member == ""):
                    continue

                #get owlready object of member
                member_owlready = self.getOWLObject(member)

                #if the member is a property, update addConcern relation
                if(self.isProperty(member_owlready) == True):

                    member_owlready.addConcern.append(RHSNode.owlreadyObj)


                #update formula's members
                #negated in formula
                if(member[0] == "-"):

                    print(member, " is negated")

                    new_Formulas.includesMember.append(member_owlready)
                    member_owlready.negMemberOf.append(new_Formulas)

                #not negated in formula
                else:

                    new_Formulas.includesMember.append(member_owlready)
                    member_owlready.memberOf.append(new_Formulas)

        #have final formula address the concern
        if last_Formulas != None:

            last_Formulas.formulasAddConcern.append(RHSNode.owlreadyObj)

    #updating names
    def editPropertyName(self,node,new_name):

        node.owlreadyObj.name = new_name

    def editFormulaName(self,node,new_name):

        node.owlreadyObj.name = new_name

    def editComponentName(self,node,new_name):

        node.owlreadyObj.name = new_name

    #switching polarity from regular to negated
    def switchToRegMemberOf(self,relationChild,relationParent):

        relationChild.owlreadyObj.memberOf.append(relationParent.owlreadyObj)
        relationChild.owlreadyObj.negMemberOf.remove(relationParent.owlreadyObj)

    #switching polarity from negative to regular
    def switchToNegMemberOf(self,relationChild,relationParent):

        relationChild.owlreadyObj.negMemberOf.append(relationParent.owlreadyObj)
        relationChild.owlreadyObj.memberOf.remove(relationParent.owlreadyObj)

    #remove relations
    def removePropertyAddressesConcernRelation(self,parent,child):

        child.owlreadyObj.addConcern.remove(parent.owlreadyObj)

    def removeConcernFormulasRelation(self,parent,child):

        child.owlreadyObj.formulasAddConcern.remove(parent.owlreadyObj)

    def removeFormulasFormulasRelation(self,parent,child):

        child.owlreadyObj.memberOf.remove(parent.owlreadyObj)
        parent.owlreadyObj.includesMember.remove(child.owlreadyObj)

    def removeFormulasPropertyRelations(self,parent,child):

        child.owlreadyObj.memberOf.remove(parent.owlreadyObj)
        parent.owlreadyObj.includesMember.remove(child.owlreadyObj)

    def removeRelatedToRelation(self,parent,child):

        child.owlreadyObj.relateToProperty.remove(parent.owlreadyObj)

    #find the concern addressed by a formula
    def findAddressedConcern(self,owlreadyobj):

        addressed_concern = owlreadyobj.formulasAddConcern

        if (len(addressed_concern) > 0):

            self.addressed_concerns = addressed_concern
            return addressed_concern

        else:

            if(len(owlreadyobj.memberOf) == 0):
                return

            self.findAddressedConcern(owlreadyobj.memberOf[0])

    #find owlNode given a name
    def getOwlNode(self,name):

        #loop through nodeArray and find matching object
        for node in self.nodeArray:

            if(node.name == name):

                return node

        print("couldnt find from app " + name)
        return 0

    #find owlready object given name
    def getOWLObject(self,name):

        #remove negation
        if name[0] == "-":
            name = name[1:]

        #search for object in owlready ontology, could return multiple that match pattern
        obj_list = self.owlreadyOntology.search(iri = "*" + name)

        #couldnt find anything if the length is 0
        if(len(obj_list) == 0):
            print("couldnt find " + name)


        obj_names = []

        #remove namespaces from all possible objects
        for obj in obj_list:
            obj_names.append(remove_namespace(obj))

        i = 0

        #find object that matches exact name
        while i < len(obj_names):

            if(obj_names[i] == name):
                obj = obj_list[i]
                break

            i = i + 1

        return obj

    #searches base ontology for owlNode
    def getOwlNodeFromBase(self,name):

        for node in self.owlBase.allConcerns_owlNode:

            if(node.name == name):
                return node

        return 0

    #updates quantity values for different types
    def setNumbers(self):

        try:
            self.numComponents = len(self.allComponents_owlNode)
        except:

            self.numComponents = 0

        try:
            self.numProperties = len(self.allProperties_owlNode)
        except:
            self.numProperties = 0
        self.numNodes = self.numComponents + self.numProperties

    #determines whether an owlready object is a property or not
    def isProperty(self,owlreadyobj):

        #gets type
        is_a = owlreadyobj.is_a

        #sees if one of its type is a property
        for istype in is_a:

            if(remove_namespace(istype) == "Property"):

                return True

        return False

    #sees if an owlready object is a conjunction or disjunction or neither
    def getSubType(self,owlreadynode):

        types = owlreadynode.is_a

        if(self.owlreadyOntology.Conjunction in types ):

            return "Conjunction"

        elif(self.owlreadyOntology.Disjunction in types):

            return "Disjunction"

        else:

            return "None"
