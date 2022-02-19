## National Institute of Standards and Technology's Cyber-physical System Framework

As part of this project, I performed a variety of work surrounding the National Institute of Standards and Technology's (NIST) Cyber-physical System (CPS) Framework. The CPS Framework is a knowledge representation and reasoning approach to modeling and constraining a CPS's abilities, shortcomings and behavior. At its core is a tree-like, heiarchial knowledge structure called an ontology. In this tree structure, each node is a concern stakeholders care about relating to a CPS. At the highest level of the ontology are the broadest concerns, called aspects and include things like Trustworthiness, Business, Data, Human etc. Below these aspects are series of concerns, which in turn can have sub-concerns creating a tree with depth ranging from 2-6 depending on the branch. Properties or requirements the CPS takes on help to address these concerns, leading to a concern's satisfaction or dissastisfaction. In general, for a concern to be satisfied all of its recursive sub-concerns must also be satisfied. For example, in order for the Security concern to be satisfied, its subconcerns Cybersecurity and Physicalsecurity must be satified, which also have their own subconcerns. With this methodology, and the use of logic porgramming, given a combination of properties or requirements a CPS takes, it can be determined which concerns associated iwth the CPS are satisfied and unsatisfied. This is the core concept of the CPS Framework, however much of my group's work aimed to sophisticate the CPS Framework. 

Much of my group's work involved theorizing and implementing more advanced reasoning tasks and general sophistication of the CPS Framework design and workflow. My most major contribution to this proect was the development of an application allowing a user to create and edit a CPS's digital represntation or ontology. I also worked heavily on incorporating Artificial-Intellgence related concerns into the CPS Framework, as these ideas were largely left out at the CPS Framework's conception. 

During this project, I was advised by New Mexico State University's Department Head Dr. Son Tran and worked in lockstep with the now Dr. Thanh Ngyuen. I also worked with parties outside NMSU, including Dr. Marcello Balduccini and some of his graduate students at St. Joseph's University as well as NIST's Associate Director for CPS, Dr. Edward Griffor. In total we wrote 4 published papers while I was working on this project, all of which I contributed to and are included in this repo.

### Ontology Editor Application

My largest and most independent portion of this project was the creation of a graphical user interface which allows users to visually create and modify a CPS's digital representation, or ontology. I developed this application from the ground up, and learned a lot in the process. It is written in Python and makes use of several powerful packages which helped me out quite a bit. The power in this application is how much it simplifies working with ontologies, which are stored in large, convoluted text files with wonky notations. With the application, a user can simply load in an ontology by giving a file name, modify the ontology by clicking on nodes in the tree and using buttons, and export the ontology when they are all done. The code itself is written in python and is over 4,000 lines long, making use of object-oriented programming with classes capturing individual concepts such as concerns, ontologies, graphs etc. I wrote extensive documentation for this tool to help new users, which can be found in this repo.


<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/NIST_CPS_Framework_Research/README_images/ontology_editor.png?raw=true" width="1000" height="600">
</p>

#### Operation Supported

**Empty Space Clicks** - When a click on empty space is detected, the user is prompted with a window to provide a name and what type of individual they'd like to add. This information is used to create a new individual with owlready2, and the visualiztion is refreshed.

**Node Clicks** - When a user clicks on a node, they are presented with a window with options depending on the type of node they selected. From here, they can modify a node's name, delete the clicked node, or add indviduals directly connected to a node. This information is used to create a new individual with owlready2, and the visualiztion is refreshed.

**Relations Window Button** - When a user clicks the Relations button, a new window appears where they can select the parent and child nodes. If a relation between them already exists, they can remove that relation, or they can add a relation of which type is auto-detected depending on the type of the parent and child.

**Dependecies Window** - When the Dependencies button is clicked, a new entry-window is displayed where users can add more complex dependencies to address a concern. They can construct formulas in IF THEN format, where if a boolean combination of properties is satisfied, then it satisfies a given concern. Behind the scenes, these formulas are recursively constructed and added to the owlready2 ontology. 

**Rem Relationless** - Often times after many operations are performed, a user ends up with individuals not connected to any other part of the ontology. These can be removed by clicking this button. The program detects all individuals that don't have relations, and are presented to the user for confirmation of deleting. 

**Input/Output Ontology** - Users can type in names of ontology files to search for to load from, and provide a filename the currently loaded ontology should be exported to. The actual loading and saving of ontologies is handled by owlready2. 

**Hovered Node Information** - The application keeps track of the user's mouse location. If the mouse hovers over an indvidual, information about it is displayed to the user.

**Zooming/Panning** - Scroll bars are presented allowing a user to modify the matplotlib window displayed on the interface. A user can also use their scroll-wheel to expand or shrink the matplotlip window displayed. 

****

### Incorporation of AI into CPS Framework 

To help capture AI-Related concerns into the CPS Framework, we explored options for representation. While the CPS Framework already robustly represented a broad range of business, trusthworthiness and security concerns, it didn't reflect some of the more nuanced ideas associated with Artificial Intelligence. It was especially lacking in capturing the non-technical ideas of AI such as transparency in decision making, bias towards certain groups of humans and rationality in decisions it makes. These ideas all need to be represented, and require their own corresponding concerns in the CPS Framework which could be added under already existing Aspects. However, in order to keep all AI-related concerns in a common location, we decided to create a new aspect called Rationality which captures AI-related ideas. This aspect and its membering concerns require a CPS to be transparent, in compliance with AI standards, have unbiased, sound training data while maintaining privacy among other things. Our group wrote a paper describing how AI can be incorporated into the CPS Framework, of which I am the first author on. We are also in the process of wrting and handbook and journal article describing the usefullness of the CPS Framework as it relates to business processes making use of AI.  


<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/NIST_CPS_Framework_Research/README_images/rationality.png?raw=true" width="400" height="400">
</p>


### Completion of CPS Ontology Workflow
Another major contribution I made to this project was assisting in the completion of the CPS Ontology workflow. Before the editor came along, the only piece of software we had was the Java-written reasoner, which is able to reason about hardcoded ontologies with logic programming. The introduction of the editor helped to streamline the creation and modification of ontologies. We also had no way of visualizing the results of any reasoning we performed. I worked to create a module within the reasoner to export the results as a csv file for later use. Working with a graduate student from SJU, I helped to create a Tableau visualization which is better able to capture the status of a CPS after reasoning. After this work, we were able to construct ontologies, reason about them and visualize results. 


#### Edit an Ontology

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/NIST_CPS_Framework_Research/README_images/editor_2.PNG?raw=true" width="600" height="400">
</p>


#### Reason about an Ontology and Export as CSV
Performing reasoning queries to determine concern satisfaction in the CPS Framework. Dr. Thanh Ngyuen developed the shwon CPS Reasoner. I developed the support for exporting results as a CSV for future visualization. 

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/NIST_CPS_Framework_Research/README_images/reasoner.png?raw=true" width="600" height="400">
</p>

#### Visualize in Tableau
A cleaner, easier to understand visualization than the editor can provide, indicating which concerns are satisfied and which are unsatisfied. Visualization developed by Amanda Bailey from SJU. I worked with her to develop a file format, and method of creation to serve as the source of visualizations. 

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/NIST_CPS_Framework_Research/README_images/CPS_tree.png?raw=true" width="600" height="400">
</p>


