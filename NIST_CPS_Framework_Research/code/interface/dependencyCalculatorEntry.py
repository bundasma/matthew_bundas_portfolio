'''
dependencyCalculatorEntry

This class handles the actual parsing of the entry windows in the GUI, processes
it, and creates the dependency represented in the entry windows. It also handles
the instantiation and propogation of the Tkinter window object.

More extensive documentation is provided.

'''

import tkinter as tk
from tkinter import *
import numpy as np
import tkinter.font as tkFont
from parse import *
from parseDependency import parseAndCreateRules
from owlFormula import owlFormula


spartangreen = "#18453b"
class dependencyCalculatorEntry:

    def __init__(self,toBind,owlbase,owlapplication,GUI):


        self.labelFont = tkFont.Font(family = "Helvetica",size = 30, weight = "bold")

        #which window it's bound to
        self.boundTo = toBind

        #ontologies it has access to
        self.owlBase = owlbase
        self.owlApplication = owlapplication

        #GUI it belongs to
        self.GUI = GUI

        #declaration of some attributes
        self.numLHSNodes = 0
        self.numRHSNodes = 0

        #actual owlNodes in LHS/RHS
        self.LHSNodes = []
        self.RHSNodes = []

        #text in LHS/RHS
        self.showTextLHS = ""
        self.showTextRHS = ""

        #opening of Tkinter windows the dependency calculator uses
        self.ifFrame = tk.Frame(self.boundTo,bg = spartangreen)
        self.ifFrame.place(relwidth = .05,relheight = .20, relx = .02,rely = .09)

        self.LHSFrame = tk.Frame(self.boundTo, bg = spartangreen)
        self.LHSFrame.place(relwidth = .45, relheight = .30, relx = .08, rely = .05)

        self.thenFrame = tk.Frame(self.boundTo, bg = spartangreen)
        self.thenFrame.place(relwidth = .15, relheight = .20, relx = .58, rely = .09)

        self.RHSFrame = tk.Frame(self.boundTo, bg = spartangreen)
        self.RHSFrame.place(relwidth = .20, relheight = .30, relx = .75, rely = .05)

        self.leftPFrame = tk.Frame(self.boundTo, bg = "yellow")
        self.leftPFrame.place(relwidth = .05, relheight = .18, relx = 0.05, rely = .35)

        self.rightPFrame = tk.Frame(self.boundTo, bg = "black")
        self.rightPFrame.place(relwidth = .05, relheight = .18, relx = 0.12, rely = .35)

        self.andFrame = tk.Frame(self.boundTo, bg = "black")
        self.andFrame.place(relwidth = .10, relheight = .18, relx = .22, rely = .35)

        self.orFrame = tk.Frame(self.boundTo, bg = "black")
        self.orFrame.place(relwidth = .10, relheight = .18, relx = .34, rely = .35)

        self.notFrame = tk.Frame(self.boundTo, bg = "black")
        self.notFrame.place(relwidth = .10, relheight = .18, relx = .46, rely = .35)

        self.LHSEntry = Text(self.LHSFrame,width = 50, height = 4)
        self.LHSEntry.pack()
        self.LHSEntry.insert(tk.END,"(")

        self.RHSEntry = Text(self.RHSFrame, width = 50, height = 4)
        self.RHSEntry.pack()

        self.ifButton = Button(self.ifFrame, text = "IF", bg = spartangreen, font = self.labelFont, fg = "white",padx = 20, command = self.onIfClick)
        self.ifButton.pack()

        self.thenButton = Button(self.thenFrame, text = "THEN", bg = spartangreen, font = self.labelFont, fg = "white", padx = 20, command = self.onThenClick)
        self.thenButton.pack()

        self.leftPButton = Button(self.leftPFrame, text = "(", bg = spartangreen, font = self.labelFont, fg = "white", padx = 20, command = self.onLeftPClick)
        self.leftPButton.pack()

        self.rightPButton = Button(self.rightPFrame, text = ")", bg = spartangreen, font = self.labelFont, fg = "white", padx = 20, command = self.onRightPClick)
        self.rightPButton.pack()

        self.andButton = Button(self.andFrame, text = "and", bg = spartangreen, font = self.labelFont, fg = "white", padx = 20, command = self.onAndClick)
        self.andButton.pack()

        self.orButton = Button(self.orFrame, text = "or", bg = spartangreen, font = self.labelFont, fg = "white", padx = 20, command = self.onOrClick)
        self.orButton.pack()

        self.notButton = Button(self.notFrame, text = "not", bg = spartangreen, font = self.labelFont, fg = "white", padx = 20, command = self.onNotClick)
        self.notButton.pack()

        self.buttonFrame = tk.Frame(self.boundTo,bg = "grey")
        self.buttonFrame.place(relwidth = .90, relheight = .40, relx = .05, rely = .55)

        self.parseLHSB = tk.Button(self.buttonFrame, text = "ParseLHS", padx = 10, height = 1, width = 25, bg = spartangreen, fg = "white",borderwidth = 5, command = self.parseLHS)
        #self.parseLHSB.pack()

        self.parseRHSB = tk.Button(self.buttonFrame, text = "ParseRHS", padx = 10, height = 1, width = 25, bg = spartangreen, fg = "white",borderwidth = 5, command = self.parseRHS)
        #self.parseRHSB.pack()

        self.createDependencyB = tk.Button(self.buttonFrame, text = "Create Dependency", padx = 10, height = 1, width = 35, bg = spartangreen, fg = "white", borderwidth = 5, command = self.onCreateDependencyB)
        self.createDependencyB.pack()

        #default to editing LHS to start
        self.editing = self.LHSEntry

    #handles parsing of LHS text
    def parseLHS(self):

        print("trying to parse LHS")

        #get text from entry
        LHS_text = self.LHSEntry.get(1.0,END)

        #get formulas, function defined in src file
        forms = parseAndCreateRules(LHS_text,self.RHSNode.name)

        self.LHSNodes = forms

    #handles parsing of RHS text
    def parseRHS(self):

         #get RHS text
         self.RHS_text = self.RHSEntry.get(1.0,END)

         #remove spaces and newlines, as it should just be one node
         self.RHS_text = self.RHS_text.replace("\n","")
         self.RHS_text = self.RHS_text.replace(" ","")

         #finds the actual owlNode
         self.RHSNode = self.findNode(self.RHS_text)

    #when the user clicks the AND button
    def onAndClick(self):

        lhstext = self.LHSEntry.get(1.0,END)

        #if there's already a space on the end, add and with no spaces
        if(lhstext[len(lhstext) - 2] == " "):
            self.LHSEntry.insert(INSERT,"and")

        #otherwise add and with spaces
        else:
            self.LHSEntry.insert(INSERT," and ")

    #when the user clicks the OR button
    def onOrClick(self):

        lhstext = self.LHSEntry.get(1.0,END)

        #if there's already a space on the end, add or with no spaces
        if(lhstext[len(lhstext) - 2] == " "):
            self.LHSEntry.insert(INSERT,"or")

        #otherwise add or with spaces
        else:
            self.LHSEntry.insert(INSERT," or ")

    #when the user licks the NOT button
    def onNotClick(self):

        lhstext = self.LHSEntry.get(1.0,END)

        #if there's already a space on the end, add not with no spaces
        if(lhstext[len(lhstext) - 2] == " "):
            self.LHSEntry.insert(INSERT,"not")

        #otherwise add not with spaces
        else:
            self.LHSEntry.insert(INSERT," not ")

    #when user clicks ( button, insert a (
    def onLeftPClick(self):

        self.LHSEntry.insert(INSERT,"(")

    #when user clicks ) button, insert a )
    def onRightPClick(self):

        self.LHSEntry.insert(INSERT,")")

    #finds the owlNode with the given name
    def findNode(self,name):

        #search base ontology for name
        for node in self.owlBase.allConcerns_owlNode:

            if(node.name == name):
                return node

        #search application ontology for name
        for node in self.owlApplication.nodeArray:

            if(node.name == name):
                return node


        print("couldn't find " + str(name))
        return 0

    #when user clicks create dependency button, parse and add it
    def onCreateDependencyB(self):

        #parse text
        self.parseRHS()
        self.parseLHS()

        #add dependency
        self.owlApplication.addNewDependency(self.LHSNodes,self.RHSNode)

        self.GUI.updateTree()

    #change editing to LHS
    def onIfClick(self):
        self.editing = self.LHSEntry

    #change editing to RHS
    def onThenClick(self):
        self.editing = self.RHSEntry
