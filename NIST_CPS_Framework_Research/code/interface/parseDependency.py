'''
parseDependency

Contains supplemental functions for handling of dependency creation. Creating dependencies
is tricky, as you need to parse text, and recursively create formulas of certain
types to build up the dependency. These functions are used in the dependencyCalculatorEntry
and stored here.

More extensive documentation is provided.

'''

from parse import *
import numpy as np
import re
from owlFormula import owlFormula


#from https://stackoverflow.com/questions/4284991/parsing-nested-parentheses-in-python-grab-content-by-level
def parenthetic_contents(string):
    """Generate parenthesized contents in string as pairs (level, contents)."""
    stack = []
    for i, c in enumerate(string):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            yield (len(stack), string[start + 1: i])



#tests whether input has nested parentheses
def contains_nested(in_line):

    #doesnt have a ( and a )
    if(in_line.find(")") != -1 and in_line.find("(") != -1):

        return True

    else:
        return False


#gets members of a formula that aren't nested in multiple sets of parentheses
def get_nonnested_members(in_line):

    #what we are searching for
    searcher = "(" + in_line + ")"

    #detect operator
    if(in_line.find("and") != -1):

        op = "and"

    elif(in_line.find("or") != -1):

        op = "or"

    else:

        op = "unknown"

    #split to find members of formula
    splitline = in_line.split(" ")

    members = []
    negated = False

    #look through each element of the formula
    for element in splitline:

        if element == "and" or element == "or":
            continue

        #found not, make sure next member we find is counted as negated
        elif (element == "not"):

            negated = True

        #append members to list
        else:

            if(negated == True):

                members.append("-" + element)
                negated = False

            else:
                members.append(element)


    return members, op, searcher



def searchForAndReplaceNested(current_formulas,in_line):

    for formula in current_formulas:

        in_line = in_line.replace(formula.searcher,formula.name)


    return in_line

#sort based on first element
def sortFunc(e):

    return e[0]

#handles parsing of LHS and creating of rules to address RHS
def parseAndCreateRules(text,RHS_name):


    formlist = []

    #counter to make unique formulas
    rulenum = 1

    #find formulas
    forms = list(parenthetic_contents(text))

    #sort formulas
    forms.sort(key = sortFunc,reverse = True)

    #for each formula, create condition, starting with non-nested formulas
    for form in forms:

        line = form[1]

        #if there's no nested, directly make the formula
        if(contains_nested(line) == False):

            newformula = owlFormula()

            #get info about formula
            newformula.members, newformula.operator, newformula.searcher = get_nonnested_members(line)

            #generate formula name
            newformula.name = RHS_name + "_Condition_" + str(rulenum)

            rulenum += 1

            #add new formula to list
            formlist.append(newformula)

        #replace nested formula with non-nested
        else:

            x = searchForAndReplaceNested(formlist,line)

            #get info about formula
            newformula = owlFormula()

            #get info about formula
            newformula.members, newformula.operator, newformula.searcher = get_nonnested_members(x)

            newformula.name = RHS_name + "_Condition_" + str(rulenum)

            rulenum += 1

            #add new formula to list
            formlist.append(newformula)



    return formlist
