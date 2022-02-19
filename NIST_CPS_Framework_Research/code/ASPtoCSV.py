'''
exportASP

This script provides the infrastructure to export results from the ASP Reasoner
to a CSV file. It does so with two classes, csvNode and ASPExporter. csvNode is
the unit block representing a single individual involved in the ASP output. ASPExporter
handles parsing the file, constructing csvNodes, assigning their attributes, and
ultimately outputting the csv file. It essentially works by re-constucting
a tree-like structure capturing the output of the ASP Reasoner, and uses this
struture to write the CSV file easier.

More documentation is provided.

'''



from parse import *
import csv
import argparse

import os


class csvNode:

    def __init__(self):

       self.name = None #name of individual
       self.type = None #type of individual
       self.parents = [] #parents of individual
       self.children = [] #children of individual
       self.lineage = [] #contains itself and all connected nodes above it
       self.aspectTree = None #the aspect the individual belongs to
       self.numVisited = 0

       self.level = 1 #level of tree individual belongs to
       self.sum = 0 #used for vis, essentially the node's footprint, how many descendants are leaf nodes
       self.satisfied = "Satisfied" #whether individual is satisfied or not

       self.leaf = False #whether node is a leaf of tree or not


class ASPExporter:

    #takes a filepath as input, does all operations up to actual exportation
    def __init__(self,filename):

        self.file = None #filename it is exporting
        self.allLines = None #lines of raw file

        #lists of nodes
        self.allNodes = []
        self.allAspects_Node = []
        self.allConcerns_Node = []
        self.allSubConcernRelations_str_str = []
        self.allLeaves_Node = []

        #output information
        self.outputRows = []
        self.outputFields = None

        #highest level in tree
        self.maxLevel = 0

        #handles all operations to parse, export as CSV
        self.loadFile(filename)
        self.addAllNodes()
        self.splitDoubleParented()
        self.assignDataMembers()
        self.removeRelationless()


    #loads the file in, assigns the line which we need to parse
    def loadFile(self,filename):

        self.file = open(filename,mode = "r")
        self.allLines = self.file.readlines()
        self.encodingLine = self.allLines[4]


    #adds in all nodes by parsing the encodingLine, sets their child/parent relation as well as satisfaction
    def addAllNodes(self):

        self.addAllAspects()
        self.addAllConcerns()
        self.addAllSubconcernRelations()
        self.addAllChildrenParents()
        self.setSatisfaction()


    #handles setting data members like level,sum, lineage etc
    def assignDataMembers(self):

        #set level,aspectTree,leaf
        for node in self.allNodes:

            self.assignLevel(node,node)
            self.assignAspectTree(node,node)
            self.assignLeaf(node)

        #sum and lineage depend on whether leaf status, so they have to go after
        for node in self.allNodes:
            self.assignSum(node,node)
            self.assignLineage(node,node)


        #pre determined max level for easier Tableau usage
        self.maxLevel = 10

        #add to max level how many columns are to the left of concern columns
        self.lineLength = self.maxLevel + 6

        #create a list of empty strings of specific length, used as template
        self.masterLine = []

        for i in range(self.lineLength):

            self.masterLine.append("")


    #nodes with multiple parents need to be split into individuals with one parent
    def splitDoubleParented(self):

        #check all nodes to see if they have more than 1 parent
        for node in self.allNodes:

            #if they have more than one parent, for each extra parent create a node for them
            if(len(node.parents) > 1):

                #for each extra parent, create a new node
                for i in range(1,len(node.parents)):

                    #get extra parent
                    parent = node.parents[i]

                    #remove extra parent from original node's parent list
                    node.parents.remove(parent)

                    #create new csvNode
                    newNode = csvNode()

                    #it has one parent, the extra parent we are taking care of
                    newNode.parents = [parent]

                    #same attributes as oringal node
                    newNode.children = node.children.copy()
                    newNode.type = node.type
                    newNode.name = node.name
                    newNode.satisfied = node.satisfied

                    self.allNodes.append(newNode)

                    if(newNode.type == "Concern"):
                        self.allConcerns_Node.append(newNode)

                    if(newNode.type == "Aspect"):
                        self.allAspects_Node.append(newNode)



    #parses the encodingLine for aspects
    def addAllAspects(self):

        #find aspects
        parsed_aspects = ' '.join(r[0] for r in findall("aspect(\"cpsf:{}\")", self.encodingLine))

        #separate names into list
        allAspects_strList = parsed_aspects.split(" ")

        #create csvNode for each aspect
        for aspect in allAspects_strList:

            newCSVNode = csvNode()

            newCSVNode.name = aspect
            newCSVNode.type = "Aspect"

            self.allNodes.append(newCSVNode)
            self.allAspects_Node.append(newCSVNode)

    #parses the encodingLine for concerns
    def addAllConcerns(self):

        #find concerns
        parsed_concerns = ' '.join(r[0] for r in findall(" concern(\"cpsf:{}\")", self.encodingLine))

        #separate names into list
        allConcerns_strList = parsed_concerns.split(" ")

        #create csvNode for each aspect
        for concern in allConcerns_strList:

            #check to see if we already created a node for this name (probably as aspect)
            if(self.getCSVNode(concern) != 0):
                continue


            newCSVNode = csvNode()

            newCSVNode.name = concern
            newCSVNode.type = "Concern"

            self.allNodes.append(newCSVNode)
            self.allConcerns_Node.append(newCSVNode)


    #parses encodingLine for suboncern relations
    def addAllSubconcernRelations(self):

        #parse subconcern relations

        #get nodes that are first member in relation
        subconcerns_parse = ' '.join(r[0] for r in findall(" subconcern(\"cpsf:{}\",\"cpsf:{}\"", self.encodingLine))

        #get nodes that are second member in relation
        subconcerns_parse2 = ' '.join(r[1] for r in findall(" subconcern(\"cpsf:{}\",\"cpsf:{}\"", self.encodingLine))

        #split names into list
        subconcerns_list = subconcerns_parse.split(" ")
        subconcerns_list2 = subconcerns_parse2.split(" ")

        for i in range( len(subconcerns_list)):

              parent = subconcerns_list[i]
              child = subconcerns_list2[i]

              self.allSubConcernRelations_str_str.append([parent,child])



    #parses encoding line for satisfied or not satisfied
    def setSatisfaction(self):

        #finds names of nodes that are satisfied
        parsed_satisfied =  ' '.join(r[0] for r in findall(" h(sat(\"cpsf:{}\"),0)", self.encodingLine))

        #finds names of nodes that are unsatisfied
        parsed_unsatisfied =  ' '.join(r[0] for r in findall(" -h(sat(\"cpsf:{}\"),0)", self.encodingLine))

        #creates list of names of satisfied/unsatisfied
        allSatisfied_strList = parsed_satisfied.split(" ")
        allUnSatisfied_strList = parsed_unsatisfied.split(" ")

        #sets satisfied attribute for each satisfied node
        for satisfied in allSatisfied_strList:

            if(satisfied == "all"):
                continue

            satisfied_node = self.getCSVNode(satisfied)

            if(satisfied_node == 0):
                continue

            satisfied_node.satisfied = "Satisfied"

        #sets satisfied attribute for each unsatisfied node
        for unsatisfied in allUnSatisfied_strList:

            if(unsatisfied == "all"):# or unsatisfied == "roles"):
                continue

            unsatisfied_node = self.getCSVNode(unsatisfied)

            if(unsatisfied_node == 0):
                continue

            unsatisfied_node.satisfied = "Unsatisfied"


    #sets children and parent relations based on parsed subconcern relations
    def addAllChildrenParents(self):

        #for each relation gathered, set parent/child attributes
        for relation in self.allSubConcernRelations_str_str:

            #get parents of relations
            parent_name = relation[0]
            parent_node = self.getCSVNode(parent_name)

            if(parent_node == 0):
                continue

            #get children of relations
            child_name = relation[1]
            child_node = self.getCSVNode(child_name)

            if(child_node == 0):
                continue

            #set parent/children attributes
            parent_node.children.append(child_node)
            child_node.parents.append(parent_node)



    #recursive function to set level of passed node
    def assignLevel(self,ognode,node):

        #if there's no parents (reached top), stop
        if(len(node.parents) == 0):
            return

        #if theres a parent, jump to it, increment level
        else:
            ognode.level += 1
            self.assignLevel(ognode,node.parents[0])

    # function to set leaf status of passed node
    def assignLeaf(self,node):

        #if node has no children, it's a leaf
        if(len(node.children) == 0):
            node.leaf = True
            self.allLeaves_Node.append(node)

        else:
            node.leaf = False


    #recursive function to set sum for passed node, which is essentially how many leave children the node has
    def assignSum(self,ognode,currentnode):

        #set sum for leaf node
        if(currentnode.leaf == True):

            ognode.sum = ognode.sum + 1
            return

        #keep parsing until we find a leaf
        for child in currentnode.children:

            self.assignSum(ognode,child)


    #recursive function to assign lineage to passed node, which is itself and ordered parent nodes
    def assignLineage(self,ognode,node):

        #insert add parent as lineage
        ognode.lineage.insert(0,node)

        #if there's no parents, we've parsed to top, completed lineage
        if(len(node.parents) == 0):
            return

        #continue parsing up tree
        self.assignLineage(ognode,node.parents[0])


    #recursive function to set aspectTree for passed node, which is what aspect tree the node belongs t
    def assignAspectTree(self,ognode,node):

        #if there are no parents, we are at top, it's an aspect, set attribute
        if(len(node.parents) == 0):

            ognode.aspectTree = node
            return

        #if there are still parents, continue parsing
        else:
            self.assignAspectTree(ognode,node.parents[0])


    #handles performing output to passed output file path
    def doOutput(self,output_name):

        self.setFields()
        self.sortNodes()
        self.setOutputRows()
        self.writeRows(output_name)

    #set column headers
    def setFields(self):

        self.outputFields = self.masterLine.copy()
        self.outputFields[0] = "CPS_Aspect"

        for i in range(1,self.maxLevel):
            self.outputFields[i] = "CPS_Concern" + str(i)


        self.outputFields[self.lineLength - 6] = "Level"
        self.outputFields[self.lineLength - 5] = "Pad"
        self.outputFields[self.lineLength - 4] = "To Pad"
        self.outputFields[self.lineLength - 3] = "Sum"
        self.outputFields[self.lineLength - 2] = "Satisfied"
        self.outputFields[self.lineLength - 1] = "Aspect Tree"


    #each leaf node gets proccessed and outputted to csv
    def setOutputRows(self):

        for node in self.allLeaves_Node:

            #loop through lineage
            for i in range(len(node.lineage)):

                line = self.masterLine.copy()

                #loop through current node and its lineage
                for j in range(0,i + 1):

                    line[j] = node.lineage[j].name

                    if(j == i):

                        currnode = node.lineage[j]

                        line[self.lineLength - 6] = str(currnode.level)
                        line[self.lineLength - 5] = "1"
                        line[self.lineLength - 4] = "1"
                        line[self.lineLength - 3] = str(currnode.sum)
                        line[self.lineLength - 2] = str(currnode.satisfied)
                        line[self.lineLength - 1] = str(currnode.aspectTree.name)
                        self.outputRows.append(line)

        for node in self.allLeaves_Node:

            #loop through lineage
            for i in range(len(node.lineage)):

                line = self.masterLine.copy()

                #loop through current node and its lineage
                for j in range(0,i + 1):


                    line[j] = node.lineage[j].name

                    if(j == i):

                        currnode = node.lineage[j]

                        line[self.lineLength - 6] = str(currnode.level)
                        line[self.lineLength - 5] = "203"
                        line[self.lineLength - 4] = "203"
                        line[self.lineLength - 3] = str(currnode.sum)
                        line[self.lineLength - 2] = str(currnode.satisfied)
                        line[self.lineLength - 1] = str(currnode.aspectTree.name)
                        self.outputRows.append(line)



    #writes the output we constructed
    def writeRows(self,out_name):

        with open(out_name, 'w',newline = '') as csvfile:
            #creating a csv writer object
            csvwriter = csv.writer(csvfile)

            #writing the fields
            csvwriter.writerow(self.outputFields)

            #writing the data rows
            csvwriter.writerows(self.outputRows)


    #removes nodes which don't have any children/parents, if there are any
    def removeRelationless(self):

        numremoved = 0

        for node in self.allNodes:

            #if there's no parents/children, get rid of it
            if(len(node.children) == 0 and len(node.parents) == 0):

                self.allNodes.remove(node)
                numremoved+=1


        for node in self.allLeaves_Node:

             #if there's no parents/children, get rid of it
             if(len(node.children) == 0 and len(node.parents) == 0):

                self.allLeaves_Node.remove(node)
                numremoved+=1

        for node in self.allAspects_Node:

             #if there's no parents/children, get rid of it
             if(len(node.children) == 0 and len(node.parents) == 0):

                self.allAspects_Node.remove(node)
                numremoved+=1

        for node in self.allConcerns_Node:

             #if there's no parents/children, get rid of it
             if(len(node.children) == 0 and len(node.parents) == 0):

                self.allConcerns_Node.remove(node)
                numremoved+=1

        if(numremoved != 0):

            self.removeRelationless()

    #returns node with given name and specified parent
    def getCSVNodeWithParent(self,name,parent):

        for node in self.allNodes:

            if(node.name == name and node.parents[0] == parent):

                return node

        return 0

    #returns node with specified name
    def getCSVNode(self,name):

         for node in self.allNodes:

             if(node.name == name):
                 return node


         return 0

    #key function for sorting based on aspectTree
    def byAspectTree(self,node):

        return node.aspectTree.name

    #sorts leaves based on the aspect tree they belong to
    def sortNodes(self):
        self.allLeaves_Node.sort(key = self.byAspectTree)


#driver function which inputs given ASP file, outputs CSV with given filepath
def exportCSV(input_encoding_str,output_file_path):

    print("loading from ")
    exporter = ASPExporter(input_encoding_str)
    print("successfully loaded")

    print("exporting to ", output_file_path)
    exporter.doOutput(output_file_path)
    print("successfully exported")




if __name__ == "__main__":

    print("called main")

    parser = argparse.ArgumentParser()

    parser.add_argument("ASP_input_filename",type = str)
    parser.add_argument("CSV_output_filename",type = str)

    args = parser.parse_args()

    print("ARGUMENTS:")
    print("first = ", args.ASP_input_filename)
    print("second = ", args.CSV_output_filename)
    print()

    exportCSV(args.ASP_input_filename,args.CSV_output_filename)
