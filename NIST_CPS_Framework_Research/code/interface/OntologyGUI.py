'''
OntologyGUI

This class is the master class of the application. It handles the production
of the user interface using Tkinter and handling of iteractions with the interface.
It allows a user to load and output an ontology. It makes use of the package owlready2
to handle some basic ontology operations, and calls outside classses and functions
for others.

More extensive documentation is provided.

'''




import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from owlFunctions import is_asp_or_conc
import tkinter as tk
import tkinter.font as tkFont
from tkinter import *
from script_networkx import remove_namespace
from dependencyCalculatorEntry import dependencyCalculatorEntry
from owlBase import owlBase
from owlApplication import owlApplication
from owlGraph import owlGraph
import platform
from owlready2 import *

spartangreen = "#18453b"

class OntologyGUI:

    def __init__(self,master):


        self.fontStyleTest = tkFont.Font(family="Lucida Grande", size=80, weight = "bold")

        self.zoom = 1
        self.zoomIndex = 105

        #boolean flags denoting which windows are open for proper event handling
        self.lcWindowOpen = False
        self.rcWindowOpen = False
        self.relationWindowOpen = False
        self.dependencyWindowOpen = False
        self.removeConfirmationWindowOpen = False
        self.removeChildrenWindowOpen = False
        self.polarityWindowOpen = False
        self.RLIWindowOpen = False


        #which node is currently being hovered over
        self.hoveredNode = None

        #coordinates of mouse events
        self.eventX = None
        self.eventY = None

        #boolean flags for understanding which ontologies have been loaded
        self.owlBaseLoaded = False
        self.owlAppLoaded = False

        #attribute for instance of owlApplication class
        self.owlApplication = None

        self.allTreeNodes = None


        self.master = master
        self.operatingSystem = platform.system()

        self.buttonWidth = 20
        self.outputButtonWidth = 15

        #tkinter looks different on different OS, handle them accordingly
        if(self.operatingSystem == "Windows"):

            print("Dealing with Windows")

            self.fontsize = 12

            self.fontsize_0_5 = 6
            self.fontsize_1 = 12
            self.fontsize_2 = 22
            self.fontsize_3 = 34

            self.buttonFontColor = "white"
            self.buttonBGColor = spartangreen

        #non-windows OS
        else:

            print("Dealing with Non-Windows OS")

            self.fontsize = 8.5

            self.fontsize_0_5 = 6
            self.fontsize_1 = 8.5
            self.fontsize_2 = 16
            self.fontsize_3 = 24

            self.buttonFontColor = "black"
            self.buttonBGColor = "grey"

        #call handle zoom when middle mouse is used
        self.master.bind("<Button-4>", self.handleZoom)
        self.master.bind("<Button-5>", self.handleZoom)
        self.master.bind("<MouseWheel>", self.handleZoom)

        master.title("Ontology GUI")

        #set up main canvas
        self.canvas = Canvas(master, height = 1200, width = 1900, bg = "#18453b")
        self.canvas.pack()

        #set up title text
        self.masterHeaderFrame = Frame(master,bg ="#18453b" )
        self.masterHeaderFrame.place(relwidth = .8, relheight = .06, relx = .1, rely = 0.01)

        #set up header text
        self.masterHeaderText = Label(self.masterHeaderFrame, text="CPS Ontology Editor",fg = "white",bg = "#18453b", font = "Helvetica 30 bold italic")
        self.masterHeaderText.pack()

        #set up footer text
        self.footerFrame = Frame(master,bg ="#18453b")
        self.footerFrame.place(relwidth = .4, relheight = .10, relx = .61, rely = 0.98)
        self.footerText = Label(self.footerFrame, text="Matt Bundas, Prof. Son Tran, Thanh Ngyuen, Prof. Marcello Balduccini",fg = "white",bg = "#18453b", font = "Helvetica 8 bold italic", anchor = "e")
        self.footerText.pack()

        #set up frame on left for inputs
        self.leftControlFrame = Frame(master, bg="white")
        self.leftControlFrame.place(relwidth = .2, relheight = .91, relx = .01, rely = 0.07)

        #set up prompt/entry for input ontology
        self.inputBasePrompt = Label(self.leftControlFrame, text = "Input base ontology", font = promptFont,fg= "#747780",bg = "white")
        self.inputBasePrompt.pack()

        self.inputBaseEntry = Entry(self.leftControlFrame, width = 30,borderwidth = 5,highlightbackground="white", fg = "#18453b",font = entryFont)
        self.inputBaseEntry.pack()
        self.inputBaseEntry.insert(0,"cpsframework-v3-base-development.owl")

        #button to load ontology, calls function which handles loading
        self.loadBaseOntologyB = tk.Button(self.leftControlFrame, text = "Load Base Ontology",padx = 10, pady = 1, width = self.buttonWidth, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5,font = buttonFont,command = self.loadBaseOntology)
        self.loadBaseOntologyB.pack()

        self.addSpace(self.leftControlFrame,"white","tiny")

        #input onotology entry
        self.inputAppPrompt = Label(self.leftControlFrame, text = "Input application ontology", font = promptFont,fg= "#747780",bg = "white")
        self.inputAppPrompt.pack()

        self.inputAppEntry = Entry(self.leftControlFrame, width = 30,borderwidth = 5,highlightbackground="white", fg = "#18453b",font = entryFont)
        self.inputAppEntry.pack()
        self.inputAppEntry.insert(0,"cpsframework-v3-sr-LKAS-Configuration-V1.owl")

        #button to load ontology, calls function which handles loading
        self.loadAppOntologyB = tk.Button(self.leftControlFrame, text = "Load Application Ontology",padx = 10, pady = 1, width = self.buttonWidth,bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5,font = buttonFont,command = self.loadAppOntology)
        self.loadAppOntologyB.pack()

        self.unloadAppOntologyB = tk.Button(self.leftControlFrame, text = "Unload Application Ontology")

        self.addSpace(self.leftControlFrame,"white","tiny")


        #sets up prompt/entry for name of output owl file
        self.outputPrompt = Label(self.leftControlFrame, text =  "Output Base Name", font = promptFont,fg= "#747780",bg = "white")
        self.outputPrompt.pack()

        self.outputEntry = Entry(self.leftControlFrame, width = 30,borderwidth = 5,highlightbackground="white", fg = "#18453b",font = entryFont)
        self.outputEntry.pack()
        self.outputEntry.insert(2, "cpsframework-v3-base-development.owl")

        #sets up button to call function which handles saving ontology
        self.saveOntologyB = tk.Button(self.leftControlFrame, text = "Output Base Ontology",padx = 10, pady = 1, width = self.outputButtonWidth, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5,font = buttonFont, command = self.saveOntology)
        self.saveOntologyB.pack()

        self.addSpace(self.leftControlFrame,"white","tiny")

        #sets up prompt/entry for name of output owl file
        self.outputAppPrompt = Label(self.leftControlFrame, text =  "Output App Name", font = promptFont,fg= "#747780",bg = "white")
        self.outputAppPrompt.pack()

        self.outputAppEntry = Entry(self.leftControlFrame, width = 30, borderwidth = 5, highlightbackground = "white", fg = "#18453b", font = entryFont)
        self.outputAppEntry.pack()
        self.outputAppEntry.insert(2, "cpsframework-v3-sr-LKAS-Configuration-V1.owl")

        self.saveAppOntologyB = tk.Button(self.leftControlFrame, text = "Output App Ontology",padx = 10, pady = 1, width = self.outputButtonWidth,bg = "#18453b", fg = self.buttonFontColor, borderwidth = 5, font = buttonFont, command  = self.saveAppOntology)
        self.saveAppOntologyB.pack()

        self.addSpace(self.leftControlFrame,"white","tiny")

        self.saveOntologyLaunchASPB = tk.Button(self.leftControlFrame, text = "Output Ontology and Run ASP", padx = 10, pady = 5, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.outputAndLaunchASP)
        #self.saveOntologyLaunchASPB.pack()
        self.addSpace(self.leftControlFrame,"white","tiny")

        #sets up gray box for information window
        self.infoFrame = Frame(self.leftControlFrame, bg = "#747780", bd = 5 )
        self.infoFrame.place(relwidth = .9, relheight = .50, relx = .05, rely = .44)

        #information about ontology
        self.infoFrameHeaderLabel = Label(self.infoFrame,text = "Ontology Information", font = headerFont, fg = "white", bg = "#747780")
        self.infoFrameHeaderLabel.pack()
        self.owlInfoFrame = Frame(self.infoFrame, bg = spartangreen, bd = 5)
        self.owlInfoFrame.place(relwidth = .94, relheight = .45, relx = .03, rely = .06)
        self.owlInfoFrame.update()

        #information about hovered node
        self.indInfoHeaderFrame = Frame(self.infoFrame, bg = "#747780",bd = 5)
        self.indInfoHeaderFrame.place(relwidth = .9, relheight = .07, relx  = .05, rely = .525)
        self.indInfoHeaderLabel = Label(self.indInfoHeaderFrame,text = "Hovered Information", font = headerFont, fg = "white", bg = "#747780")
        self.indInfoHeaderLabel.pack()
        self.indInfoFrame = Frame(self.infoFrame,  bg = spartangreen, bd = 5)
        self.indInfoFrame.place(relwidth = .94, relheight = .41, relx = .03, rely = .58)


        #text values for ontology information
        self.owlBaseNameText = tk.StringVar()
        self.owlBaseNameText.set("Base")

        self.owlAppNameText = tk.StringVar()
        self.owlAppNameText.set("App")

        self.totalNodeText = tk.StringVar()
        self.totalNodeText.set("Total Nodes")

        self.numAspectsText = tk.StringVar()
        self.numAspectsText.set("Num Aspects")

        self.numConcernsText = tk.StringVar()
        self.numConcernsText.set("Num Concerns")

        self.numPropertiesText = tk.StringVar()
        self.numPropertiesText.set("Num Properties")

        self.numComponentsText = tk.StringVar()
        self.numComponentsText.set("Num Components")

        self.owlBaseNameInfo = Label(self.owlInfoFrame, textvariable =  self.owlBaseNameText, font = "Monaco 12 bold" ,fg= "white",bg = spartangreen)
        self.owlBaseNameInfo.pack()

        self.owlAppNameInfo = Label(self.owlInfoFrame, textvariable =  self.owlAppNameText, font = "Monaco 12 bold",fg= "white",bg = spartangreen)
        self.owlAppNameInfo.pack()

        self.numNodesInfo = Label(self.owlInfoFrame, textvariable =  self.totalNodeText, font = infoFont,fg= "white",bg = spartangreen)
        self.numNodesInfo.pack()

        self.numAspectsInfo = Label(self.owlInfoFrame, textvariable =  self.numAspectsText, font = infoFont,fg= "white",bg = spartangreen)
        self.numAspectsInfo.pack()

        self.numConcernsInfo = Label(self.owlInfoFrame, textvariable =  self.numConcernsText, font = infoFont,fg= "white",bg = spartangreen)
        self.numConcernsInfo.pack()

        self.numPropertiesInfo = Label(self.owlInfoFrame, textvariable =  self.numPropertiesText, font = infoFont,fg= "white",bg = spartangreen)
        self.numPropertiesInfo.pack()

        self.numComponentsInfo = Label(self.owlInfoFrame, textvariable =  self.numComponentsText, font = infoFont,fg= "white",bg = spartangreen)
        self.numComponentsInfo.pack()

        #text values for hovered node information
        self.indNameText = tk.StringVar()
        self.indNameText.set("Name")

        self.indTypeText = tk.StringVar()
        self.indTypeText.set("Type")

        self.indParentText = tk.StringVar()
        self.indParentText.set("Parent Name")

        self.indChildrenText = tk.StringVar()
        self.indChildrenText.set("Children")

        self.indRelPropertiesText = tk.StringVar()
        self.indRelPropertiesText.set("Relevant Properties")

        self.indNameInfo = Label(self.indInfoFrame, textvariable = self.indNameText ,fg= "white",bg = spartangreen,font = infoFont)
        self.indNameInfo.pack()

        self.indTypeInfo = Label(self.indInfoFrame, textvariable = self.indTypeText,fg= "white",bg = spartangreen,font = infoFont)
        self.indTypeInfo.pack()

        self.indParentInfo = Label(self.indInfoFrame, textvariable =  self.indParentText,fg= "white",bg = spartangreen,font = infoFont)
        self.indParentInfo.pack()

        self.indChildInfo = Label(self.indInfoFrame, textvariable =  self.indChildrenText,fg= "white",bg = spartangreen,font = "Monaco 12 bold")
        self.indChildInfo.pack()

        self.indPropertyInfo = Label(self.indInfoFrame, textvariable =  self.indRelPropertiesText,fg= "white",bg = spartangreen,font = infoFont)
        self.indPropertyInfo.pack()


        #sets up gray box to put text to show info about most recent operation
        self.textBoxFrame = Frame(self.leftControlFrame,bg = "#747780", bd = 5)
        self.textBoxFrame.place(relwidth = .9, relheight = .04, relx = .05 ,rely = .95)

        self.summaryText = tk.StringVar()
        self.summaryText.set("")
        self.summaryLabel = Label(self.textBoxFrame, textvariable = self.summaryText, font = summaryFont,fg= "white",bg = "#747780")
        self.summaryLabel.pack()

        #sets up frame for ontology tree to exist
        self.treeFrame = tk.Frame(self.master, bg="white")
        self.treeFrame.place(relwidth = .75, relheight = .91, relx = .22, rely = 0.07)
        self.treeFig, self.treeAxis = plt.subplots(figsize = (15,15))
        self.treeChart = FigureCanvasTkAgg(self.treeFig,self.treeFrame)
        self.treeAxis.clear()
        self.treeAxis.axis('off')
        self.treeChart.get_tk_widget().pack()

        #connects clicks to handleClick function
        self.treeChart.mpl_connect("button_press_event",self.handleClick)

        #connects general mouse movement to handleHover function
        self.treeChart.mpl_connect("motion_notify_event",self.handleHover)


        #set up sliders in tree frame
        self.xSliderFrame = tk.Frame(self.treeFrame,bg = "white")
        self.xSliderFrame.place(relwidth = .7, relheight = .05, relx = .15, rely = .95)

        self.xSliderScale = Scale(self.xSliderFrame, from_ = 0, to = 100,orient = HORIZONTAL,bg = "gray", fg = "white",length = 900,command = self.scale_tree)
        self.xSliderScale.pack()

        self.ySliderFrame = tk.Frame(self.treeFrame,bg = "white")
        self.ySliderFrame.place(relwidth = .03, relheight = .7, relx = .95, rely = .15)

        self.ySliderScale = Scale(self.ySliderFrame, from_ = 80, to = 0,orient = VERTICAL,bg = "gray", fg = "white",length = 900,command = self.scale_tree)
        self.ySliderScale.pack()

        #set up various buttons in main frame
        self.relationButtonFrame = tk.Frame(self.treeFrame,bg = "white")
        self.relationButtonFrame.place(relwidth = .08, relheight = .05, relx = .01, rely = .01)

        self.relationB = tk.Button(self.relationButtonFrame, text = "Relations",width = 10, padx = 10, pady = 5, bg = spartangreen, fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.onRelationButton)
        self.relationB.pack()

        self.dependenciesButtonFrame = tk.Frame(self.treeFrame, bg = "white")
        self.dependenciesButtonFrame.place(relwidth = .08, relheight = .05, relx = .01, rely = .07)

        self.dependenciesB = tk.Button(self.dependenciesButtonFrame, text = "Dependencies", width = 10, padx = 10, pady = 5, bg = spartangreen, fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.onDependencyButton)
        self.dependenciesB.pack()

        self.remremoveChildrenFrame = tk.Frame(self.treeFrame,bg = "white")
        self.remremoveChildrenFrame.place(relwidth = .10, relheight = .05, relx = .89, rely = .01)

        self.removeremoveChildrenB = tk.Button(self.remremoveChildrenFrame, text = "Rem Relationless",padx = 10, pady = 5, bg = spartangreen, fg = self.buttonFontColor,borderwidth = 5, font = buttonFont,  command = self.removeFloaters)
        self.removeremoveChildrenB.pack()


    #loads the specified base ontology file in
    def loadBaseOntology(self):

        #directory to load from
        load_dir_name = "workshop_ontologies/"

        #instantitates owlBase with given entry name
        self.owlBase = owlBase(load_dir_name + self.inputBaseEntry.get())

        #construct and propogate summary message
        summary = "Loaded base ontology " + "file://./" + self.inputBaseEntry.get()
        self.summaryText.set(summary)

        #flag for operation handling
        self.owlBaseLoaded = True

        #refresh tree visualization
        self.updateTree()

    #loads the specified base ontology file in
    def loadAppOntology(self):

        #directory to load from
        load_dir_name = "workshop_ontologies/"

        #instantitates owlApplication with given entry name
        self.owlApplication = owlApplication(load_dir_name + self.inputAppEntry.get(),self.owlBase)

        #construct and propogate summary message
        summary = "Loaded application ontology " + "file://./application_ontologies" + self.inputAppEntry.get()
        self.summaryText.set(summary)

        #flag for operation handling
        self.owlAppLoaded = True

        #refresh tree visualization
        self.updateTree()

    #saves the base ontology in rdf format
    def saveOntology(self):

        #directory to save to
        save_dir_name = "workshop_ontologies/"

        #name to save to
        output_file = self.outputEntry.get()

        #save base ontology
        self.owlBase.owlReadyOntology.save(file = save_dir_name + "out" + output_file, format = "rdfxml")

        #construct and propogate summary message
        summary = "Outputted Base ontology to file: " + output_file
        self.summaryText.set(summary)

    #saves the app ontology in rdf format
    def saveAppOntology(self):

        #do noothing if we don't have an application to save
        if(self.owlApplication == None):
            return

        #directory to save to, directory where reasoner loads from
        save_dir_name = "../../src/asklab/querypicker/QUERIES/EX5-sr-lkas/"

        #name to save to
        output_file = self.outputAppEntry.get()

        #save application ontology
        self.owlApplication.owlreadyOntology.save(file =  save_dir_name + output_file, format = "rdfxml")

        #construct and propogate summary message
        summary = "Outputted Application ontology to file: " + output_file
        self.summaryText.set(summary)

    #creates master list of tree nodes from base and application
    def setAllTreeNodes(self):

        if(self.owlBaseLoaded == True):

            self.allTreeNodes = self.owlBase.allConcerns_owlNode

        if(self.owlAppLoaded == True):

            self.allTreeNodes.extend(self.owlApplication.nodeArray)

    #checks if an concern with the passed name exists
    def check_existence(self,name):

        for node in self.allTreeNodes:

            if(node.name == name):
                return True

        return False

    #clears the tree, re-sets up owlBase and graph, draws it
    def updateTree(self):

         #clear current graph
         self.treeAxis.clear()
         self.treeAxis.axis('off')

         #re-instantiate nodes in base ontology
         self.owlBase.initializeOwlNodes()

         #reset number of nodes, aspects concerns in base
         self.owlBase.setNumbers()

         #if we have an app loaded
         if(self.owlApplication != None):

             #re-instantiate nodes in base ontology
             self.owlApplication.initializeOwlNodes()
             self.owlApplication.setNumbers()

         #create owlGraph class instance
         self.owlTree = owlGraph(self.owlBase,self.owlApplication)

         #scales tee given zoom, max size
         self.scale_tree(3)

         #draw the tree
         self.owlTree.draw_graph(self.treeAxis,self.fontsize)
         self.treeChart.draw()

         #update text showing stats
         self.updateOwlStats()


         self.setAllTreeNodes()

    #changes zoom level, fontsize, then calls updateTree to draw it again
    def handleZoom(self,event):

        #event is object containing info on mouse/button event
        #zoomIndex is continuous value changed by scrolling in/out

        original_zoom = self.zoom

        #zooming out, reduce zoomIndex
        if event.num == 4 or event.delta < 0:

            if(self.zoomIndex - 1 >= 90):
                self.zoomIndex = self.zoomIndex - 1

        #zooming in, increase zoomIndex
        if event.num == 5 or event.delta > 0:

            if(self.zoomIndex + 1 <= 130):
                self.zoomIndex = self.zoomIndex + 1

        #set zoom level and fontsize depending on current zoomIndex value
        if(self.zoomIndex >= 90 and self.zoomIndex < 100):
            self.zoom = .5
            self.fontsize = self.fontsize_0_5

        elif(self.zoomIndex >= 100 and self.zoomIndex < 110):
            self.zoom = 1
            self.fontsize = self.fontsize_1

        elif(self.zoomIndex >= 110 and self.zoomIndex < 120):
            self.zoom = 2
            self.fontsize = self.fontsize_2

        elif(self.zoomIndex >= 120 and self.zoomIndex < 130):
            self.zoom = 3
            self.fontsize = self.fontsize_3

        #update tree if zoom level changed
        if(original_zoom != self.zoom):
            self.updateTree()




    #changes the portion of axis we view according to zoom level, slider position
    def scale_tree(self,var):

         #get minimum x and y values
         leftmostx = self.owlTree.minX + self.owlTree.totalX*self.xSliderScale.get()/100
         leftmosty = self.owlTree.minY + self.owlTree.totalY*self.ySliderScale.get()/100

         #adjust x and y values depending on zoom level
         if(self.zoom == .5):

             leftmosty = self.owlTree.minY - .5*self.owlTree.totalY
             rightmosty = self.owlTree.maxY + .5 * self.owlTree.totalY

             rightmostx = leftmostx + .50 * self.owlTree.totalX

         if (self.zoom == 1):

             leftmosty = self.owlTree.minY
             rightmostx = leftmostx + .20*self.owlTree.totalX
             rightmosty = self.owlTree.maxY

         if (self.zoom == 2):

             rightmostx = leftmostx + .10*self.owlTree.totalX
             rightmosty = leftmosty + .60*self.owlTree.totalY

         if (self.zoom == 3):

             rightmostx = leftmostx + .05*self.owlTree.totalX
             rightmosty = leftmosty + .25*self.owlTree.totalY

         #set axis values (viewing window) given calculated values
         self.treeAxis.set(xlim=(leftmostx, rightmostx), ylim=(leftmosty, rightmosty))

         #get window sizes
         rmx = rightmostx - leftmostx
         rmy = rightmosty - leftmosty

         #get XYRatio, used to handle mouse hovering stuff
         self.XYRatio = rmx/rmy

         #draw the tree with new axes
         self.treeChart.draw()

    #finds y distance where mouse is considred to be hovering over a node
    def getYLimit(self):

        if(self.zoom == .5):
            return 35
        elif(self.zoom == 1):
            return 20
        elif(self.zoom == 2):
            return 18
        elif(self.zoom == 3):
            return 12

    #finds x distance where mouse is considred to be hovering over a node
    def getXLimit(self,name):

        base = 4.20*len(name) + 8.6

        if(self.zoom == .5):
            return base*1.95

        elif(self.zoom == 1):

            return base

        elif(self.zoom == 2):

            return base * (.90)

        elif(self.zoom == 3):

            return base * (.75)


    #calculates distance between node and latest event, for hovering/click events
    def getDistance(self,node):

         #get position of node in tree
         try:
             nodepos = self.owlTree.graphPositions[node.name]

         except:
             return np.inf

         nodeposx = nodepos[0]
         nodeposy = nodepos[1]*1.0

         #calculate distance between event and node
         distance = np.sqrt((self.eventX - nodeposx)**2 + ((self.eventY - nodeposy)*self.XYRatio)**2)

         return distance


    #takes a mouse event, returns the closest node to the mouse event
    def getNearest(self,event):

        #get event coordinates
        self.eventX = event.xdata
        self.eventY = event.ydata

        #sort nodes based on distance
        self.allTreeNodes.sort(key = self.getDistance)

        #get closest nodes
        closest = self.allTreeNodes[0]
        secondclosest = self.allTreeNodes[1]
        thirdclosest = self.allTreeNodes[2]

        #get x/y distances for closest nodes
        closestx = np.abs(self.owlTree.graphPositions[closest.name][0] - self.eventX)
        closesty = np.abs(self.owlTree.graphPositions[closest.name][1] - self.eventY)*self.XYRatio

        secondclosestx = np.abs(self.owlTree.graphPositions[secondclosest.name][0] - self.eventX)
        secondclosesty = np.abs( self.owlTree.graphPositions[secondclosest.name][1] - self.eventY)*self.XYRatio

        thirdclosestx = np.abs(self.owlTree.graphPositions[thirdclosest.name][0] - self.eventX)
        thirdclosesty = np.abs(self.owlTree.graphPositions[thirdclosest.name][1] - self.eventY)*self.XYRatio

        #try to return one of three nodes if they are within x/y event limits
        if(closestx < self.getXLimit(closest.name) and closesty < self.getYLimit()):

            #print("returning from first ", closest.name)
            return closest

        elif(secondclosestx < self.getXLimit(secondclosest.name) and secondclosesty < self.getYLimit()):

            #print("returning from second ",secondclosest.name)
            return secondclosest

        elif(thirdclosestx < self.getXLimit(thirdclosest.name) and thirdclosesty < self.getYLimit()):

            #print("returning from third ",thirdclosest.name)
            return thirdcloses

        else:

            #print("returning none")
            return None


    #handles mouse hovering, throws away nonsense events, updates concern info window
    def handleHover(self,event):

        #do nothing if no base is loaded
        if(self.owlBaseLoaded == False):
            return

        NoneType = type(None)

        #if event doesn't have enough data, discard it
        if(type(event.xdata) == NoneType or type(event.ydata) == NoneType):
            return

        #get nearest node
        nearest_node = self.getNearest(event)

        #if closest nodes aren't within limits (getNearest returns None), reset hovering text
        if(nearest_node == None):


            self.indNameText.set("Name")
            self.indTypeText.set("Type")
            self.indParentText.set("Parent Name")
            self.indChildrenText.set("Children")
            self.indRelPropertiesText.set("Relevant Properties")

            self.hoveredNode = None

            return

        #if the nearest node isn't already handled, update hovering information
        if(nearest_node != self.hoveredNode):


            self.hoveredNode = nearest_node
            self.indNameText.set("Name - " + str(self.hoveredNode.name))
            self.indTypeText.set("Type - " + str(self.hoveredNode.type))

            parentString = "Parents -"

            #handle parent node text formatting
            i = 1
            for parent in self.hoveredNode.parents:

                if(i%2 == 0):
                    parentString = parentString + "\n" + parent.name

                else:
                    parentString = parentString + " " + parent.name

                i = i + 1

            self.indParentText.set(parentString)

            #handle children node text formatting
            childString = "Children -"
            nchildren = len(self.hoveredNode.children)

            if(nchildren >= 10):
                divisor = 3
            else:
                divisor = 2

            i = 1
            for child in self.hoveredNode.children:

                if(len(child.name) >= 25):

                    childString = childString + "\n" + child.name + "\n"

                elif(i % divisor == 0):
                    childString = childString + "\n" + child.name

                else:
                    childString = childString + " " + child.name

                i = i + 1

            self.indChildInfo.configure(font = "Monaco " + str(self.getChildFS(nchildren)) + " bold")
            self.indChildrenText.set(childString)
            self.indRelPropertiesText.set("Relevant Properties - ")

    #gets fontsize depending on how many children we need to print
    def getChildFS(self,n):

        if(n >= 10):
            return 7

        elif(n >= 8):
            return 9

        elif(n >= 6):
            return 10
        elif(n >= 4):
            return 11
        else:
            return 12


    #gets fontsize depending on text size
    def getCorrFS(self,text):

        n = len(text)

        if (n > 43):
            return 7
        elif (n > 37):
            return 9
        elif (n > 33):
            return 10
        elif (n > 30):
            return 11
        else:
            return 12



    #updates the global ontology stats according to numbers stored in owlBase
    def updateOwlStats(self):

            self.totalNodes = self.owlBase.numNodes
            text = "Base-" + str(self.owlBase.owlName)
            basefs = self.getCorrFS(text)

            if(basefs <= 8):

                textlen = int(len(text)/2)
                text = text[ : textlen] + "\n" + text[textlen : ]
                basefs = 10



            self.owlBaseNameInfo.configure(font = "Monaco " + str(basefs) + " bold")
            self.owlBaseNameText.set(text)

            self.numAspectsText.set("Num Aspects - " + str(self.owlBase.numAspects))
            self.numConcernsText.set("Num Concerns - "  + str(self.owlBase.numConcerns))

            if(self.owlAppLoaded == True):

                textapp = "App-" + str(self.owlApplication.owlName)

                appfs = self.getCorrFS(textapp)

                if(appfs <= 8):

                    textlen = int(len(textapp)/2)

                    textapp = textapp[ : textlen] + "\n" + textapp[textlen : ]



                    appfs = 10

                self.owlAppNameInfo.configure(font = "Monaco " + str(appfs) + " bold")

                self.owlAppNameText.set(textapp)


                self.numPropertiesText.set("Num Properties - " + str(self.owlApplication.numProperties))
                self.numComponentsText.set("Num Components - " + str(self.owlApplication.numComponents))

                self.totalNodes = self.totalNodes + self.owlApplication.numNodes

            self.totalNodeText.set("Num Nodes - " + str(self.totalNodes))

    #handles clicks depending on which button is clicked, which windows are open
    def handleClick(self,event):

        #left click
        if(event.button == 1):


            if(self.relationWindowOpen == True):
                self.handleRelationLeftClick(event)

            elif(self.dependencyWindowOpen == True):
                self.handleDependencyClick(event)
            else:
                self.onLeftClick(event)

        #right clicks
        elif(event.button == 3):

            if(self.dependencyWindowOpen == True):

                self.handleDependencyClick(event)
            else:
                self.onRightClick(event)

    #handling of left click when click is on empty space
    def handleEmptyLeftClick(self):

        #dont do anything if there is no base ontology or the window is already open
        if(self.RLIWindowOpen == True or self.owlBaseLoaded == False):

            print(self.RLIWindowOpen)
            print(self.owlBaseLoaded)
            print("returning")
            return

        self.errorDisplayed = False

        #open relationless individual window
        self.RLIWindow = tk.Toplevel(height = 500, width = 400, bg = spartangreen)
        self.RLIWindow.transient(master = self.master)
        self.RLIWindowOpen = True
        self.RLIWindow.title("Add Individual")

        #set closing protocal
        self.RLIWindow.protocol("WM_DELETE_WINDOW",self.RLIWindowClose)

        #set up relationless individual window, frames, texts, buttons
        self.RLIWindowHeaderFrame = tk.Frame(self.RLIWindow,bg = spartangreen)
        self.RLIWindowHeaderFrame.place(relwidth = .7, relheight = .12, relx = .15, rely = .01)

        self.RLIWindowHeaderText = Label(self.RLIWindowHeaderFrame, text = "Add New \nIndividual",font = headerFont, fg = "white", bg = spartangreen)
        self.RLIWindowHeaderText.pack()

        self.RLIButtonFrame = Frame(self.RLIWindow, bg = "white")
        self.RLIButtonFrame.place(relwidth = .7, relheight = .7, relx = .15, rely = .15)

        self.RLIPrompt = Label(self.RLIButtonFrame, text = "Name of New Node", font = promptFont, fg = "#747780", bg = "white")
        self.RLIPrompt.pack()

        self.RLIEntry = Entry(self.RLIButtonFrame, width = 30, borderwidth = 5, highlightbackground = "white", fg = spartangreen,font = entryFont)
        self.RLIEntry.pack()
        self.RLIEntry.insert(1,"NewNode")

        self.addSpace(self.RLIButtonFrame,"white","tiny")

        addAspectB = tk.Button(self.RLIButtonFrame, text = "Add Aspect",padx = 5, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15 ,font = buttonFont, command = self.addRLAspect)
        addAspectB.pack()

        self.addSpace(self.RLIButtonFrame,"white","tiny")

        addConcernB = tk.Button(self.RLIButtonFrame, text = "Add Concern", padx = 5, bg = "#18453b", fg = self.buttonFontColor, borderwidth = 5, width = 15, font = buttonFont, command = self.addRLConcern)
        addConcernB.pack()

        self.addSpace(self.RLIButtonFrame,"white","tiny")

        addPropertyB = tk.Button(self.RLIButtonFrame, text = "Add Property",padx = 5, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15 ,font = buttonFont, command = self.addRLProperty)
        addPropertyB.pack()

        self.addSpace(self.RLIButtonFrame,"white","tiny")

        addCompB = tk.Button(self.RLIButtonFrame, text = "Add Component",padx = 5, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.addRLComponent)
        addCompB.pack()

    #handles adding new relationless aspect
    def addRLAspect(self):

        self.owlBase.addNewAspect(self.RLIEntry.get())
        self.updateTree()

    #handles adding new relationless concern
    def addRLConcern(self):

        self.owlBase.addNewRLConcern(self.RLIEntry.get())
        self.updateTree()

    #handles adding relationless property
    def addRLProperty(self):

        self.owlApplication.addRLProperty(self.RLIEntry.get())
        self.updateTree()

    #handles adding relationless impact rule (outdated)
    def addRLIR(self):

        self.launchPolarityWindow()
        self.addIRPolarity()
        self.addingRelation = "RLIR"

    #handles adding relationless component
    def addRLComponent(self):

        self.owlApplication.addNewComponent(self.RLIEntry.get())
        self.updateTree()

    #handles closing of relationless window
    def RLIWindowClose(self):

        print("called close")

        self.RLIWindow.destroy()
        self.RLIWindowOpen = False


    #handles all left click scenarios
    def onLeftClick(self,event):

        print("got left click")

        #do nothing if window is already open, or have no base ontology
        if(self.lcWindowOpen == True or self.owlBaseLoaded == False):

            print("not doing anything")
            return

        self.errorDisplayed = False

        #get the node you just clicked on
        closestnode = self.getNearest(event)

        #if there's no nearby node, treat it as empty left click
        if(closestnode == None):

            self.handleEmptyLeftClick()
            return

        #there is a nearby node, make it self.leftClicked
        self.leftClicked = closestnode

        #which class of node is clicked on
        type_item = self.leftClicked.type

        #open up window and frames
        self.lcWindow = tk.Toplevel(height = 500, width = 400,bg = spartangreen )
        self.lcWindow.transient(master = self.master)
        self.lcWindow.title("Concern Editor")

        #flag for other event handling
        self.lcWindowOpen = True

        #protocol for when window is closed
        self.lcWindow.protocol("WM_DELETE_WINDOW", self.leftclickWindowClose)

        #set up window text, buttons etc.
        self.lcWindowHeaderFrame = tk.Frame(self.lcWindow,bg = spartangreen)
        self.lcWindowHeaderFrame.place(relwidth = .7, relheight = .12, relx = .15, rely = .01)
        self.lcButtonFrame = tk.Frame(self.lcWindow, bg = "white")
        self.lcButtonFrame.place(relwidth = .7, relheight = .7, relx = .15, rely = .15)

        self.lcWindowHeaderText = tk.StringVar()
        self.lcWindowHeaderText.set(self.leftClicked.name + "\n" + self.leftClicked.type

        self.lcWindowHeaderLabel = Label(self.lcWindowHeaderFrame, textvariable = self.lcWindowHeaderText ,fg= "white",bg = spartangreen,font = headerFont)
        self.lcWindowHeaderLabel.pack()

        self.indivNamePrompt = Label(self.lcButtonFrame, text = "Name of New Node", font = promptFont,fg= "#747780",bg = "white")
        self.indivNamePrompt.pack()

        self.indivNameEntry = Entry(self.lcButtonFrame, width = 30,borderwidth = 5,highlightbackground="white", fg = "#18453b",font = entryFont)
        self.indivNameEntry.pack()
        self.indivNameEntry.insert(1,"NewNode")

        self.addSpace(self.lcButtonFrame,"white","tiny")

        #handle different types of nodes accordingly, with different options
        if(is_asp_or_conc(self.leftClicked.type) == True):

            self.concernLeftClick(event)

        elif(self.leftClicked.type == "Component"):

            self.componentLeftClick(event)

        elif(self.leftClicked.type == "Property"):

            self.propertyLeftClick(event)

        elif(self.leftClicked.type == "Formulas"):

            self.formulaLeftClick(event)

        elif(self.leftClicked.type == "DecompositionFunction"):

            self.decompFuncLeftClick(event)

        else:

            print("Not sure what type that was")
            return

    #handles when a concern is left clicked, allow adding subconcern, parent concern, property
    def concernLeftClick(self,event):


        #set up buttons for operations
        addConcern = tk.Button(self.lcButtonFrame, text = "Add Subconcern",padx = 5, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.addConcern)
        addConcern.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

        addParent = tk.Button(self.lcButtonFrame, text = "Add Parent Concern",padx = 5, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15 ,font = buttonFont, command = self.addParent)
        addParent.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

        addPropertyB = tk.Button(self.lcButtonFrame, text = "Add Property",padx = 5, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.addPropertyAsChildofConcern)
        addPropertyB.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

        editName = tk.Button(self.lcButtonFrame, text = "Edit Name",padx = 5, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.editConcern)
        editName.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

        removeConcernB = tk.Button(self.lcButtonFrame, text = "Delete",padx = 5, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.removeConcern)
        removeConcernB.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")



    #adds a concern to the ontology, uses gui entries for inputs, updates tree afterwards
    def addConcern(self):

        #grab name of new concern
        new_concern_name = self.indivNameEntry.get()

        #handle error message, if new concern with name already exists, don't add another
        if (self.handleErrorMessage(new_concern_name,"lc") == 0):
            return

        #adds new concern
        self.owlBase.addNewConcern(new_concern_name,self.leftClicked.name)

        #prints text in textBoxFrame to tell what happend
        summary = "Added concern " + new_concern_name + " to ontology"

        self.summaryText.set(summary)

        #refresh the tree and owlBase
        self.updateTree()

        #reset the error message because we just did a successful operation
        self.handleErrorMessageDeletion("lc")


    #function to add parent to clicked node
    def addParent(self):

        #get name from Entry
        new_parent_name = self.indivNameEntry.get()

        #handle error message, if new concern with name already exists, don't add another
        if (self.handleErrorMessage(new_parent_name,"lc") == 0):
            return

        #add new concern as parent
        self.owlBase.addConcernAsParent(new_parent_name,self.leftClicked.name)

        #print summary pessage
        summary = "Added " + new_parent_name + " as Parent of " + self.leftClicked.name
        self.summaryText.set(summary)

        #clear out any error message, as we just did a good operation
        self.handleErrorMessageDeletion("lc")

        self.updateTree()

    #adds property as child of a concern
    def addPropertyAsChildofConcern(self):

        #get property name from user
        new_property_name = self.indivNameEntry.get()

        #if property already exists, present error
        if (self.handleErrorMessage(new_property_name,"lc") == 0):
            return

        #add property
        self.owlApplication.addPropertyAsChildofConcern(self.leftClicked,new_property_name)

        #clear out any error message, as we just did a good operation
        self.handleErrorMessageDeletion("lc")

        self.updateTree()

    #handles editing a concern's name
    def editConcern(self):

        #get concern names
        new_name = self.indivNameEntry.get()
        old_name = self.leftClicked.name
        ind_type = self.leftClicked.type

        #if new concern name already exists, throw error
        if (self.handleErrorMessage(new_name,"lc") == 0):
            return

        #change name of olwready object
        self.owlBase.editName(self.leftClicked,new_name)

        #print summary of operation
        summary = "Changed name of " + old_name + " to " + new_name
        self.summaryText.set(summary)

        self.updateTree()

        #refresh window's concern name, as we just changed it
        self.lcWindowHeaderText.set(new_name + "\n" + ind_type)

        #clear out any error message, as we just did a good operation
        self.handleErrorMessageDeletion("lc")

    #handles removing concern left clicked on
    def removeConcern(self):

        #check if there will be removeChildren if you delete node
        nodesChildren = self.owlBase.getChildren(self.leftClicked)

        #if it wouldn't create any removeChildren, just delete it
        if(len(nodesChildren) == 0):

            self.owlBase.removeConcern(self.leftClicked)
            self.updateTree()

            #print summary
            summary = "Removed " + self.leftClicked.name
            self.summaryText.set(summary)

            #close window, just finished operation
            self.leftclickWindowClose()

        #if deletion would create removeChildren, ask if they want to delete all of the removeChildren
        else:

            self.removeChildrenWindow = tk.Toplevel(height = 250, width = 600, bg = spartangreen)

            #removes lcwindow from screen
            self.lcWindow.withdraw()

            #open up remove children window
            self.removeChildrenWindowOpen = True
            self.removeChildrenWindow.title("Remove Children")

            self.removeChildrenWindow.protocol("WM_DELETE_WINDOW",self.removeChildrenWindowClose)

            self.removeChildrenWindowHeaderFrame = tk.Frame(self.removeChildrenWindow,bg = spartangreen)
            self.removeChildrenWindowHeaderFrame.place(relwidth = .7, relheight = .1, relx = .15, rely = .01)

            self.removeChildrenButtonFrame = tk.Frame(self.removeChildrenWindow, bg = "white")
            self.removeChildrenButtonFrame.place(relwidth = .7, relheight = .825, relx = .15, rely = .13)

            self.removeChildrenWindowHeaderText = tk.StringVar()
            self.removeChildrenWindowHeaderText.set("Choose a Deletion Option")

            self.removeChildrenWindowHeaderLabel = Label(self.removeChildrenWindowHeaderFrame, textvariable = self.removeChildrenWindowHeaderText ,fg= "white",bg = spartangreen,font = headerFont)
            self.removeChildrenWindowHeaderLabel.pack()

            self.addSpace(self.removeChildrenButtonFrame,"white","tiny")

            deleteSelectedB = tk.Button(self.removeChildrenButtonFrame, text = "Delete Selected Only", bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, padx = 5, height = 1, width = 40,font = buttonFont, command = self.removeSingleConcern)
            deleteSelectedB.pack()
            self.addSpace(self.removeChildrenButtonFrame,"white","tiny")

            deleteSelectedRelationlessChildrenB = tk.Button(self.removeChildrenButtonFrame, text = "Delete Selected + Resulting Relationless Children",padx = 5, height = 1, width = 40, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.handleRemoveIndAndRelationless)
            deleteSelectedRelationlessChildrenB.pack()

            self.addSpace(self.removeChildrenButtonFrame,"white","tiny")

            deleteSelectedAllChildrenB = tk.Button(self.removeChildrenButtonFrame, text = "Delete Selected + All Children", bg = "#18453b",padx = 5, height = 1, width = 40, fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.handleRemoveAllChildren)
            deleteSelectedAllChildrenB.pack()

            self.addSpace(self.removeChildrenButtonFrame,"white","tiny")

            cancelB = tk.Button(self.removeChildrenButtonFrame, text = "Cancel",padx = 5, height = 1, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.cancelDelete)
            cancelB.pack()

            self.addSpace(self.removeChildrenButtonFrame,"white","tiny")

    #handles when a property is left clicked, can essentially only edit name or delete it
    def propertyLeftClick(self,event):

        #set up property left click window
        editName = tk.Button(self.lcButtonFrame, text = "Edit Name",padx = 5, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.editPropertyName)
        editName.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

        removeConcernB = tk.Button(self.lcButtonFrame, text = "Delete",padx = 5, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.removeConcern)
        removeConcernB.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

    #handles changing of property name
    def editPropertyName(self):

        #get names
        new_name = self.indivNameEntry.get()
        old_name = self.leftClicked.name
        ind_type = self.leftClicked.type

        #throw error if desired name already exists
        if (self.handleErrorMessage(new_name,"lc") == 0):
            return

        #change name of olwready object
        self.owlApplication.editPropertyName(self.leftClicked,new_name)

        #print summary of operation
        summary = "Changed name of " + old_name + " to " + new_name
        self.summaryText.set(summary)

        self.updateTree()

        #clear out any error message, as we just did a good operation
        self.handleErrorMessageDeletion("lc")

        #refresh name of property in window
        self.lcWindowHeaderText.set(new_name + "\n" + ind_type)

    #handles when a formula is left clicked on
    def formulaLeftClick(self,event):

        self.addSpace(self.lcButtonFrame,"white","tiny")

        #set up buttons for operations
        addChildPropertyB = tk.Button(self.lcButtonFrame, text = "Add Child Property",padx = 5, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.addFormulaChildProperty)
        addChildPropertyB.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

        self.addSpace(self.lcButtonFrame,"white","tiny")

        editNameB = tk.Button(self.lcButtonFrame, text = "Edit Name",padx = 5, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.editFormulaName)
        editNameB.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

        removeFormulaB = tk.Button(self.lcButtonFrame, text = "Delete",padx = 5, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.removeConcern)
        removeFormulaB.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

    #adds property to a formula
    def addFormulaChildProperty(self):

        #get new property's name
        newprop_name = self.indivNameEntry.get()

        #throw error if the name already exists
        if (self.handleErrorMessage(new_property_name,"lc") == 0):
            return

        #adds property
        self.owlApplication.addNewPropertyToFormula(self.leftClicked,newprop_name)

        #clear out any error message, as we just did a good operation
        self.handleErrorMessageDeletion("lc")

        self.updateTree()

    #switches a formula to a decomposition function (outdates)
    def switchToDecompFunc(self):

        formula_owlready = self.leftClicked.owlreadyObj
        formula_owlready.is_a.append(self.owlApplication.owlreadyOntology.DecompositionFunction)
        self.updateTree()

    #handles editing of a formula name
    def editFormulaName(self):

        #get formula names
        new_name = self.indivNameEntry.get()
        old_name = self.leftClicked.name
        ind_type = self.leftClicked.type

        #throw error if desired name already exists
        if (self.handleErrorMessage(new_name,"lc") == 0):
            return

        #changes formula name
        self.owlApplication.editFormulaName(self.leftClicked,new_name)

        #print summary of operation
        summary = "Changed name of " + old_name + " to " + new_name
        self.summaryText.set(summary)

        self.updateTree()

        #update window with new name
        self.lcWindowHeaderText.set(new_name + "\n" + ind_type)

        #clear out any error message, as we just did a good operation
        self.handleErrorMessageDeletion("lc")


    #handles when a component is left clicked on
    def componentLeftClick(self,event):

        #set up buttons for operations
        addParentB = tk.Button(self.lcButtonFrame, text = "Add Parent Property",padx = 5, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.addParentProperty)
        addParentB.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

        editNameB = tk.Button(self.lcButtonFrame, text = "Edit Name",padx = 5, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.editComponent)
        editNameB.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

        removeComponentB = tk.Button(self.lcButtonFrame, text = "Delete",padx = 5, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, height = 1, width = 15, font = buttonFont, command = self.removeConcern)
        removeComponentB.pack()

        self.addSpace(self.lcButtonFrame,"white","tiny")

    #handles adding a property as parent of a node
    def addParentProperty(self):

        new_parent_name = self.indivNameEntry.get()

        #handle error message, if new concern with name already exists, don't add another
        if (self.handleErrorMessage(new_parent_name,"lc") == 0):
            return

        #add property connected to parent
        self.owlApplication.addPropertyAsParentofComponent(new_parent_name,self.leftClicked)

        #print summary message
        summary = "Added " + new_parent_name + " as Parent of " + self.leftClicked.name
        self.summaryText.set(summary)

        #clear out any error message, as we just did a good operation
        self.handleErrorMessageDeletion("lc")

        self.updateTree()


    #edit a component's Name
    def editComponent(self):

        #get names
        new_name = self.indivNameEntry.get()
        old_name = self.leftClicked.name
        ind_type = self.leftClicked.type

        #throw error if desired name already exists
        if (self.handleErrorMessage(new_name,"lc") == 0):
            return

        #change component's name
        self.owlApplication.editComponentName(self.leftClicked,new_name)

        #print summary of operation
        summary = "Changed name of " + old_name + " to " + new_name
        self.summaryText.set(summary)

        self.updateTree()

        #refresh window's name
        self.lcWindowHeaderText.set(new_name + "\n" + ind_type)

        #clear out any error message, as we just did a good operation
        self.handleErrorMessageDeletion("lc")

    #handles propogating of error message given frame and name
    def handleErrorMessage(self,new_name,frame):


        if(frame == "lc"):

            if(self.check_existence(new_name) == True):
               print("Individual Already Exists\n in Ontology")
               if(self.errorDisplayed == True):
                   self.error_message.destroy()
               self.error_message = Label(self.lcButtonFrame, text = "Individual Already Exists\n in Ontology", font = "Helvetica 8 bold italic",fg= "red",bg = "white")
               self.error_message.pack()
               self.errorDisplayed = True
               return 0

        if(frame == "rc"):
            if(self.check_existence(new_name) == True):
               print("Individual Already Exists\n in Ontology")
               if(self.rcErrorDisplayed == True):
                   self.rcerror_message.destroy()
               self.rcerror_message = Label(self.rcButtonFrame, text = "Individual Already Exists\n in Ontology", font = "Helvetica 8 bold italic",fg= "red",bg = "white")
               self.rcerror_message.pack()
               self.rcErrorDisplayed = True
               return 0


    #handles deletion of error message
    def handleErrorMessageDeletion(self,frame):

        if(frame == "lc"):

            if(self.errorDisplayed == True):
                self.error_message.destroy()
                self.errorDisplayed == False

        if(frame == "rc"):
            if(self.rcErrorDisplayed == True):
                self.rcerror_message.destroy()
                self.rcErrorDisplayed == False


    #handles closing concern editor window
    def leftclickWindowClose(self):

        self.lcWindowOpen = False
        self.leftClicked = None

        #delete all windows produced from leftclick window
        if(self.removeChildrenWindowOpen == True):
            self.removeChildrenWindowClose()

        #delete all windows produced from leftclick window
        if(self.removeConfirmationWindowOpen == True):
            self.removeConfirmationWindowClose()

        #delete left click window
        self.lcWindow.destroy()


    #handles closing removing children window
    def removeChildrenWindowClose(self):

        #update flag
        self.removeChildrenWindowOpen = False

        #bring leftclick window back
        if(self.lcWindow.state() == "withdrawn"):
            self.lcWindow.deiconify()

        #destroy remove children window
        self.removeChildrenWindow.destroy()

    #handles closing confirmation window
    def removeConfirmationWindowClose(self):

        #update flag
        self.removeConfirmationWindowOpen = False

        #bring remove chilren window back
        if(self.removeChildrenWindow.state() == "withdrawn"):
            self.removeChildrenWindow.deiconify()

        #destroy window
        self.removeConfirmationWindow.destroy()

    #handles removing of leftclicked concern
    def removeSingleConcern(self):

        self.owlBase.removeConcern(self.leftClicked)

        #close windows if they are open
        self.removeChildrenWindowClose()
        self.leftclickWindowClose()

        self.updateTree()

    #handles request of removing all of a node's children
    def handleRemoveAllChildren(self):

        #if the confirmation window is already open, do nothing
        if(self.removeConfirmationWindowOpen == True):
            return

        #gets nodes children
        all_node_children = self.owlBase.getChildren(self.leftClicked)

        #open up a confirmation window
        self.removeConfirmationWindow = tk.Toplevel(height = 700, width = 400, bg = spartangreen)

        #minimize leftclick window, remove children window
        self.lcWindow.withdraw()
        self.removeChildrenWindow.withdraw()

        #update flag
        self.removeConfirmationWindowOpen = True

        #propogate confirmation window, names of children we will remove
        self.removeConfirmationWindow.title("Remove Confirmation")
        self.removeConfirmationWindow.protocol("WM_DELETE_WINDOW",self.removeConfirmationWindowClose)

        self.removeConfirmationWindowHeaderFrame = tk.Frame(self.removeConfirmationWindow,bg = spartangreen)
        self.removeConfirmationWindowHeaderFrame.place(relwidth = .7, relheight = .05, relx = .15, rely = .01)

        self.removeConfirmationButtonFrame = tk.Frame(self.removeConfirmationWindow, bg = "white")
        self.removeConfirmationButtonFrame.place(relwidth = .7, relheight = .85, relx = .15, rely = .08)

        self.removeConfirmationWindowHeaderText = tk.StringVar()
        self.removeConfirmationWindowHeaderText.set("This operation will delete the following nodes:")

        self.removeConfirmationWindowHeaderLabel = Label(self.removeConfirmationWindowHeaderFrame, textvariable = self.removeConfirmationWindowHeaderText ,fg= "white",bg = spartangreen,font = promptFont)
        self.removeConfirmationWindowHeaderLabel.pack()

        #sets up label to add list of children to
        toDeleteLabel = Label(self.removeConfirmationButtonFrame,text = self.leftClicked.name,fg= "gray",bg = "white",font = promptFont)
        toDeleteLabel.pack()

        #add list of children
        for node in all_node_children:

            toDeleteLabel = Label(self.removeConfirmationButtonFrame,text = node.name,fg= "gray",bg = "white",font = promptFont)
            toDeleteLabel.pack()

        #introduce prompt
        self.removeConfirmationQuestionLabel = Label(self.removeConfirmationButtonFrame,text = "Would you like to Remove these?")
        self.removeConfirmationQuestionLabel.pack()


        self.addSpace(self.removeConfirmationButtonFrame,"white","tiny")

        #confirms deletion
        yesB = tk.Button(self.removeConfirmationButtonFrame, text = "Yes",padx = 10, pady = 5, width = 10, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.removeAllChildren)
        yesB.pack()

        self.addSpace(self.removeConfirmationButtonFrame,"white","tiny")

        #closes window, gets rid of deletion
        noB = tk.Button(self.removeConfirmationButtonFrame, text = "No",padx = 10, pady = 5, width = 10, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.removeConfirmationWindowClose)
        noB.pack()



    #removes all children of leftclicked node, and leftclicked node itself
    def removeAllChildren(self):

        #gets children
        all_node_children = self.owlBase.getChildren(self.leftClicked)

        #removes children one by one
        for node in all_node_children:
                print("removing " + node.name)
                self.owlBase.removeConcern(node)

        #removes concern
        self.owlBase.removeConcern(self.leftClicked)

        #close windows, update tree
        self.removeConfirmationWindowClose()
        self.removeChildrenWindowClose()
        self.leftclickWindowClose()
        self.updateTree()


    #removes the individual and chilren which would become relationless
    def removeIndAndRelationless(self):

        #retrieve children which will become relationless
        soonToBeRelationless = self.owlBase.getRelationless(self.leftClicked)

        #remove children which would become relationless
        for node in soonToBeRelationless:
                print("removing " + node.name)
                self.owlBase.removeConcern(node)

        #remove clicked node
        self.owlBase.removeConcern(self.leftClicked)

        #close all windows
        self.removeConfirmationWindowClose()
        self.removeChildrenWindowClose()
        self.leftclickWindowClose()
        self.updateTree()

    #handles confirmation of removing relationless children
    def handleRemoveIndAndRelationless(self):

        #if window is already open, do nothing
        if(self.removeConfirmationWindowOpen == True):
            return

        #get all relationless children
        all_node_relationless_children = self.owlBase.getRelationless(self.leftClicked)

        #open and propogate removing relationless window
        self.removeConfirmationWindow = tk.Toplevel(height = 600, width = 400, bg = spartangreen)
        self.removeConfirmationWindow.transient(master = self.removeChildrenWindow)

        self.removeConfirmationWindowOpen = True

        self.removeConfirmationWindow.title("Remove Confirmation")
        self.removeConfirmationWindow.protocol("WM_DELETE_WINDOW",self.removeConfirmationWindowClose)


        self.removeConfirmationWindowHeaderFrame = tk.Frame(self.removeConfirmationWindow,bg = spartangreen)
        self.removeConfirmationWindowHeaderFrame.place(relwidth = .7, relheight = .05, relx = .15, rely = .01)

        self.removeConfirmationButtonFrame = tk.Frame(self.removeConfirmationWindow, bg = "white")
        self.removeConfirmationButtonFrame.place(relwidth = .7, relheight = .85, relx = .15, rely = .08)


        self.removeConfirmationWindowHeaderText = tk.StringVar()
        self.removeConfirmationWindowHeaderText.set("This operation will delete the following nodes:")

        self.removeConfirmationWindowHeaderLabel = Label(self.removeConfirmationWindowHeaderFrame, textvariable = self.removeConfirmationWindowHeaderText ,fg= "white",bg = spartangreen,font = infoFont)
        self.removeConfirmationWindowHeaderLabel.pack()

        #add all nodes to be deleted to window
        toDeleteLabel = Label(self.removeConfirmationButtonFrame,text = self.leftClicked.name,fg= "gray",bg = "white",font = promptFont)
        toDeleteLabel.pack()

        for node in all_node_relationless_children:

            toDeleteLabel = Label(self.removeConfirmationButtonFrame,text = node.name,fg= "gray",bg = "white",font = promptFont)
            toDeleteLabel.pack()

        #add options for confirmation, or declining
        self.removeConfirmationQuestionLabel = Label(self.removeConfirmationButtonFrame,text = "Would you like to Remove these?")
        self.removeConfirmationQuestionLabel.pack()

        yesB = tk.Button(self.removeConfirmationButtonFrame, text = "Yes",padx = 10, pady = 5, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.removeIndAndRelationless)
        yesB.pack()

        noB = tk.Button(self.removeConfirmationButtonFrame, text = "No",padx = 10, pady = 5, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.removeConfirmationWindowClose)
        noB.pack()


    def cancelDelete(self):

        self.removeChildrenWindowClose()

    #takes care of right clicks, opens up window where you can add a new aspect
    def onRightClick(self,event):

        if(self.owlBaseLoaded == False or self.rcWindowOpen == True):
            return

        #set up windows and frames
        self.rcWindow = tk.Toplevel(height = 500, width = 400, bg = spartangreen)
        self.rcWindow.title("Right Click Window")
        self.rcWindow.protocol("WM_DELETE_WINDOW", self.rcWindowClose)

        #fresh window, no error to be displayed
        self.rcErrorDisplayed = False

        self.rcWindowFrame = tk.Frame(self.rcWindow,bg = spartangreen)
        self.rcWindowFrame.place(relwidth = .7, relheight = .05, relx = .15, rely = .01)

        self.rcButtonFrame = tk.Frame(self.rcWindow, bg = "white")
        self.rcButtonFrame.place(relwidth = .7, relheight = .7, relx = .15, rely = .15)

    #handles closing right click window
    def rcWindowClose(self):

        self.rcWindow.destroy()
        self.rcWindowOpen = False
        self.rightClicked = None


    #function to handle when you click Relations button, opens up window where you can do relation operations
    def onRelationButton(self):

        if(self.relationWindowOpen == True):
            return

        #set up window and frames
        self.relationWindow = tk.Toplevel(height = 500, width = 400, bg = spartangreen)
        self.relationWindow.transient(master = self.master)
        self.relationWindow.title("Add New Relation")

        self.relationWindowOpen = True
        self.readyForRelationButton = False
        self.relationClickSelecting = "Parent"

        #set deletion protocal
        self.relationWindow.protocol("WM_DELETE_WINDOW", self.relationWindowClose)

        #propogate text, buttons
        self.relationWindowFrame = tk.Frame(self.relationWindow,bg = spartangreen)
        self.relationWindowFrame.place(relwidth = .8, relheight = .20, relx = .10, rely = .01)

        showtext = "Add New Relation \nClick on Desired Parent \nThen Click on Desired Child"

        self.relationWindowHeader = Label(self.relationWindowFrame, text = showtext,fg= "white",bg = spartangreen,font = headerFont)
        self.relationWindowHeader.pack()

        self.relationButtonFrame = tk.Frame(self.relationWindow, bg = "white")
        self.relationButtonFrame.place(relwidth = .7, relheight = .7, relx = .15, rely = .20)

        self.addSpace(self.relationButtonFrame,"white","tiny")

        self.relationParentText = tk.StringVar()
        self.relationParentText.set("Relation Parent - ")

        self.relationChildText = tk.StringVar()
        self.relationChildText.set("Relation Child - ")

        self.relationParentLabel = Label(self.relationButtonFrame, textvariable = self.relationParentText, font = promptFont,fg= "#747780",bg = "white")
        self.relationParentLabel.pack()

        self.relationChildLabel = Label(self.relationButtonFrame, textvariable = self.relationChildText, font = promptFont,fg= "#747780",bg = "white")
        self.relationChildLabel.pack()


        self.addSpace(self.relationButtonFrame,"white","tiny")

        #add buttons for adding subconcern, addresses, remove relation
        addRelationB = tk.Button(self.relationButtonFrame, text = "Add Relation",padx = 10, height = 1, width = 25, bg = "#18453b",fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.addRelation)
        addRelationB.pack()

        self.addSpace(self.relationButtonFrame,"white","tiny")

        removeRelationB = tk.Button(self.relationButtonFrame, text = "Remove Relation",padx = 10, height = 1, width = 25, bg = "#18453b", fg = self.buttonFontColor,borderwidth = 5, font = buttonFont, command = self.removeRelation)
        removeRelationB.pack()

    #handles clicks to set up relations, sets either parent or child node, alternating with each click
    def handleRelationLeftClick(self,event):

        closestnode = self.getNearest(event)

        if(closestnode == None):
            return


        #if we are selecting parent, make click select parent, then have it select child next
        if(self.relationClickSelecting == "Parent"):

            #have only selected parent, not ready to create/delete a relation yet
            self.readyForRelationButton = False


            closestnode = self.getNearest(event)

            #set parent
            self.relationParent = closestnode
            self.relationParentText.set("Relation Parent - " + self.relationParent.name)

            #select child next time
            self.relationClickSelecting = "Child"

            return

        #if we are selecting child, make click select child, then have it select parent next
        elif(self.relationClickSelecting == "Child"):

            closestnode = self.getNearest(event)

            #set child
            self.relationChild = closestnode
            self.relationChildText.set("Relation Child - " + self.relationChild.name)

            #set parent next time
            self.relationClickSelecting = "Parent"

            #have selected parent and child, ready for creation/deltion of relation
            self.readyForRelationButton = True


    #adds relation between selected parent/child according to types
    def addRelation(self):

        if(self.readyForRelationButton == False):
            print("not ready for relation yet")
            return

        parent_type = self.relationParent.type
        child_type = self.relationChild.type

        #if both are concerns, add subconcern relation
        if(is_asp_or_conc(parent_type) == True and is_asp_or_conc(child_type) == True):

            self.addSubConcernRelation()

        #if parent is concern, child is property, add addressesConcern relation
        elif(parent_type == "Concern" and child_type == "Property"):

            self.addAddressesConcernRelation()

        #if parent is a concern and child is a formula, add concernFormula relation
        elif(parent_type == "Concern" and (child_type == "Formulas" or child_type == "DecompositionFunction" )):

            self.addConcernFormulaRelation()

        #if parent and chidlren are both formulas
        elif((parent_type == "Formulas" or parent_type == "DecompositionFunction") and (child_type == "Formulas" or child_type == "DecompositionFunction")):

            #if child is negated, switch to non-negated
            if(self.relationChild in self.relationParent.negChildren):

                self.owlApplication.switchToRegMemberOf(self.relationChild,self.relationParent)
                self.relationWindowClose()
                self.onRelationButton()

            #if child is not negated, switch to negated
            elif(self.relationChild in self.relationParent.children):

                self.owlApplication.switchToNegMemberOf(self.relationChild,self.relationParent)
                self.relationWindowClose()
                self.onRelationButton()

            #if relation doesn't already exist, add a non-negated relation between them
            else:
                self.addFormulaFormulaRelations()

        #if parent is formula, child is property add property to formula, or change polarity if one already exists
        elif((parent_type == "Formulas" or parent_type == "DecompositionFunction") and (child_type == "Property")):

            #if property is already negated, switch to non-negated
            if(self.relationChild in self.relationParent.negChildren):

                self.owlApplication.switchToRegMemberOf(self.relationChild,self.relationParent)
                self.relationWindowClose()
                self.onRelationButton()

            #if parent is not negated, switch to negated
            elif(self.relationChild in self.relationParent.children):

                self.owlApplication.switchToNegMemberOf(self.relationChild,self.relationParent)
                self.relationWindowClose()
                self.onRelationButton()

            #if property isn't already a part of formula, add as non-negated
            else:

                self.addFormulaPropertyRelations()

        #if parent is property, child is component, add relatedTo relation
        elif(parent_type == "Property" and child_type == "Component"):

            self.addRelatedToRelation()

        #if parent and child aren't compatible for relation, do nothing
        else:

            print("Parent and child don't make sense")
            print(parent_type)
            print(child_type)

            return

        self.updateTree()
        summary = "Added relation between " + self.relationParent.name + " and " + self.relationChild.name
        self.summaryText.set(summary)

    #adds a subconcern relation between selected parent and child
    def addSubConcernRelation(self):

        self.owlBase.addNewSubConcernRelation(self.relationParent,self.relationChild)


    def addAddressesConcernRelation(self):

        self.owlApplication.addaddConcernRelation(self.relationParent,self.relationChild)


    def addConcernFormulaRelation(self):

        self.owlApplication.addNewConcernFormulaRelation(self.relationParent,self.relationChild)


    def addFormulaFormulaRelations(self):

        self.owlApplication.addFormulaFormulaRelations(self.relationParent,self.relationChild)


    def addFormulaPropertyRelations(self):

        self.owlApplication.addFormulaPropertyRelations(self.relationParent,self.relationChild)


    def addRelatedToRelation(self):

        self.owlApplication.addNewRelatedToRelation(self.relationParent,self.relationChild)


    #removes the selected relation, at the moment just handles subconcern relation
    def removeRelation(self):

        #if parent and child both aren't selected yet, do nothing
        if(self.readyForRelationButton == False):
            print("not ready for button yet")
            return

        parent_type = self.relationParent.type
        child_type = self.relationChild.type

        #handle removing relation differently depending on types
        if(is_asp_or_conc(parent_type) == True and is_asp_or_conc(child_type) == True):

            self.removeSubConcernRelation()

        elif(parent_type == "Concern" and child_type == "Property"):

            self.removePropertyAddressesConcernRelation()

        elif(parent_type == "Concern" and (child_type == "Formulas" or child_type == "DecompositionFunction" )):

            self.removeConcernFormulaRelation()

        elif((parent_type == "Formula" or parent_type == "DecompositionFunction") and (child_type == "Formula" or child_type == "DecompositionFunction")):

            self.removeFormulaFormulaRelations()

        elif((parent_type == "Formula" or parent_type == "DecompositionFunction") and (child_type == "Property")):

            self.removeFormulaPropertyRelations()

        elif(parent_type == "Property" and child_type == "Component"):

            self.removeRelatedToRelation()

        else:

            print("Parent and child don't make sense")
            return

        self.updateTree()

        summary = "Removed relation between " + self.relationParent.name + " and " + self.relationChild.name

        self.summaryText.set(summary)

    #removes subconcern relation
    def removeSubConcernRelation(self):

        self.owlBase.removeSubConcernRelation(self.relationParent,self.relationChild)

    #remove removePropertyAddressesConcern relation
    def removePropertyAddressesConcernRelation(self):

        self.owlApplication.removePropertyAddressesConcernRelation(self.relationParent,self.relationChild)

    #remove concern/formula relation
    def removeConcernFormulaRelation(self):

        self.owlApplication.removeConcernFormulaRelation(self.relationParent,self.relationChild)

    #remove formula/formula relation
    def removeFormulaFormulaRelations(self):

        self.owlApplication.removeFormulaFormulaRelation(self.relationParent,self.relationChild)

    #remove formula/property relation
    def removeFormulaPropertyRelations(self):

        self.owlApplication.removeFormulaPropertyRelations(self.relationParent,self.relationChild)

    #remove relatedTo relation
    def removeRelatedToRelation(self):

        self.owlApplication.removeRelatedToRelation(self.relationParent,self.relationChild)


    #handles closing relations window
    def relationWindowClose(self):

        self.relationWindow.destroy()
        self.relationWindowOpen = False

    #handles when dependency button is clicked, opens window if not already
    def onDependencyButton(self):

        #if already open, do nothing
        if(self.dependencyWindowOpen == True):
             return

        #open and propogate window
        self.dependencyWindow = tk.Toplevel(height = 400, width = 1000, bg = spartangreen)
        self.dependencyWindow.transient(master = self.master)
        self.dependencyWindow.title("Add New Relation")

        self.dependencyWindowOpen = True
        self.readyForDependency = False

        #handles whether left or right side of window should be added to
        self.inserting = "left"

        self.dependencyWindowHeaderFrame = tk.Frame(self.dependencyWindow,bg = spartangreen)
        self.dependencyWindowHeaderFrame.place(relwidth = .8, relheight = .10, relx = .10, rely = .01)

        showtext = "Add New Dependency"

        #set text, buttons
        self.dependencyWindowHeader = Label(self.dependencyWindowHeaderFrame, text = showtext, fg = "white",bg = spartangreen, font = headerFont)
        self.dependencyWindowHeader.pack()

        self.dependencyWindowButtonFrame = tk.Frame(self.dependencyWindow,bg = "white")
        self.dependencyWindowButtonFrame.place(relwidth = .8, relheight = .80, relx = .10, rely = .11)


        self.dependencyEntryFrame = tk.Frame(self.dependencyWindowButtonFrame,bg = spartangreen)
        self.dependencyEntryFrame.place(relwidth = .9, relheight = .80, relx = .05, rely = .05)

        #opens dependency calculator entry
        self.DCE = dependencyCalculatorEntry(self.dependencyEntryFrame, self.owlBase, self.owlApplication,self)
        self.dependencyWindow.protocol("WM_DELETE_WINDOW",self.dependencyWindowClose)

    #handles left click when dependency window is open, gets nearest node
    #adds as "or" if right clicked, "and" if left clickced
    def handleDependencyClick(self,event):

        nearest = self.getNearest(event)

        #right click, add as or
        if event.button == 3:
            andor = "or"

        #left click, add as and
        else:
            andor = "and"

        #if not near anything, do nothing
        if(nearest == None):
            return

        #get current text we are adding to
        currentText = self.DCE.editing.get(1.0,END)

        #if space already exists on end, or is open bracket, don't start with a space
        if(currentText[len(currentText) - 2] == " " or currentText[len(currentText) - 2] == "("):

            space = ""

        #if not already spaced, start with a space
        else:

            space = " "

        #remove brackets to help processing
        currentText = currentText.replace("(","")
        currentText = currentText.replace(")","")

        #split words with space between them
        currentText = currentText.split(" ")

        goodtext = []

        #remove newlines from string
        for string in currentText:

            string = string.rstrip("\n")
            goodtext.append(string)

        goodtext = list(filter(("").__ne__, goodtext))
        goodtext = list(filter(("\n").__ne__, goodtext))

        #if inserting text is empty, insert at end
        if(len(goodtext) == 0):

            self.DCE.editing.insert(END, nearest.name)
            return

        #if current text ends with and, or, not, or ), insert just clicked name on end
        if(goodtext[-1] == "and" or goodtext[-1] == "or" or goodtext[-1] == "not" or goodtext[-1] == ")"):

            self.DCE.editing.insert(END, space + nearest.name)

        #if current text doesn't end with and/or/not/), insert clicked name and and/or
        else:
            self.DCE.editing.insert(END, space + andor + " " + nearest.name)


    #handles closing dependency window
    def dependencyWindowClose(self):

        self.dependencyWindow.destroy()
        self.dependencyWindowOpen = False


    #handles adding property as child
    def addPropertyAsChild(self):

        self.addingRelation = "PropertyAsChild"

        self.launchPolarityWindow()
        self.addConditionPolarity()
        self.addIRPolarity()

    #handles cancelling of adding property
    def cancelAddProperty(self):

        self.polarityWindowClose()


    #function to add space in specified location, with specified color and size
    def addSpace(self,on,color,size):

        if(size == "tiny"):
            self.emptySpace = Label(on, text = "", font = tiny,bg = color)

        if(size == "small"):
            self.emptySpace = Label(on, text = "", font = small,bg = color)

        if(size == "medium"):
            self.emptySpace = Label(on, text = "", font = med,bg = color)

        if(size == "large"):
            self.emptySpace = Label(on, text = "", font = large,bg = color)

        self.emptySpace.pack()

    #adds passed text to summary message
    def printSummary(self,text):

        self.summaryLabel = Label(self.textBoxFrame, text = text, font = summaryFont,fg= "white",bg = "#747780")
        self.summaryLabel.pack()

    #handles processing the output file after we save it, for now just strips string tag in owl file.
    def processFile(self,output_file):

        #read input file
        fin = open(output_file, "rt")
        #read file contents to string
        data = fin.read()
        #replace all occurrences of the required string
        data = data.replace( " rdf:datatype=\"http://www.w3.org/2001/XMLSchema#string\"", '')
        #close the input file
        fin.close()
        #open the input file in write mode
        fin = open(output_file, "wt")
        #overrite the input file with the resulting data
        fin.write(data)
        #close the file
        fin.close()


    #gets the accurate OWL Object via searching the ontology for the passed name
    def getOWLObject(self,name):

        #searches ontology for objects
        obj_list = self.owlBase.owlReadyOntology.ontology.search(iri = "*" + name)
        obj_names = []

        #removes namespace from each object, to get just individual name
        for obj in obj_list:
            obj_names.append(remove_namespace(obj))

        i = 0
        #if object is found, return it
        while i < len(obj_names):

            if(obj_names[i] == name):
                obj = obj_list[i]
                break

            i = i + 1


        return obj


    #removes all nodes which dont have any parents nor any children, and are not aspects
    def removeFloaters(self):

        self.owlBase.removeRelationless()
        self.updateTree()

        summary = "Removed all concerns with no relations"
        self.summaryText.set(summary)

    #strips namespace from text, leaving just the individual name
    def remove_namespace(in_netx):

        in_str = str(in_netx)

        #finds location of period (where real name starts)
        leng = len(in_str)
        period = leng
        for i in range(leng):
            if(in_str[i] == '.'):
                period = i
                break

        #return text after period (where real name is)
        return in_str[(period + 1):]

#start a Tkinter instance
root = Tk()

root.state("zoomed")

fontStyle = tkFont.Font(family="Lucida Grande", size=8, weight = "bold")

headerFont = tkFont.Font(family = "Helvetica",size = 14, weight = "bold")
promptFont = tkFont.Font(family = "Lucida Grande", size = 10, weight = "bold")
infoFont = tkFont.Font(family = "Monaco", size = 12, weight = "bold")
entryFont = tkFont.Font(family = "Verdana", size = 11, weight = "bold")
buttonFont = tkFont.Font(family = "Helvetica", size = 12, weight = "normal")
summaryFont = tkFont.Font(family = "Lucida Grande", size = 8, weight = "bold")

tiny = tkFont.Font(family="Lucida Grande", size=1, weight = "bold")
small = tkFont.Font(family="Lucida Grande", size=6, weight = "bold")
med = tkFont.Font(family="Lucida Grande", size=12, weight = "bold")
big = tkFont.Font(family="Lucida Grande", size=18, weight = "bold")
selectFont = tkFont.Font(family="Lucida Grande", size=11, weight = "bold")

#start instance of OntologyGUI
my_gui = OntologyGUI(root)
root.mainloop()
