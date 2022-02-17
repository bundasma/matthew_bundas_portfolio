## Simulating Enhanced C12+C12 Reaction Rates in Massive Stars During Carbon Shell Burning

As part of this project, I conducted astrophysical simulations to explore newly theorized reaction rates for a specific fusion reaction occuring in stars. 

For an overview of the project, check out the poster I created. For a more detailed look check out my paper or presentation I made as part of my senior thesis.

My favorite result of the project is a movie visualizing the simulation showing the velocity of material in roughly the inner third of the star. On the left is the simulation using the old/traditional C12/C12 fusion reaction rate, and on the right is the enhanced newly theorized reaction rates. It does a nice job showing the increased energy production due to more C12/C12 fusion. To construct this movie, I created a series of images at interval time steps using log files and python, and stiched them together using FFmpeg: https://drive.google.com/file/d/1Dik8lKlVvJzZRvcd24woERJ2HwcVFAui/view?usp=sharing

**Abstract:**
A major factor in the dynamics of stars as they pass through burning stages is how easily a given fusion reaction takes place, which is directly tied to the reaction rate. Reaction rates directly affect the composition and structure of a star which in turn affects other processes in the star. Reaction rates and in particular the Carbon-12 + Carbon-12 (C12-C12) rate are not easily measured in a laboratory setting, leading to dispute in their values. A recent paper from A. Tumino et. al experimentally found using the Trojan Horse Method (THM), underlying resonances in the C12-C12 to Mg 24 reaction rate, leading to a much enhanced rate compared to the standard Caughlan-Fowler 88 (CF88) rate. To test the potential effects of the newly found THM rate we ran two 20,000s 2-D simulations of a 25 solar mass star undergoing carbon shell burning using the hydrodynamical astrophysical code FLASH, one implementing the CF88 C12-C12 rate and one implementing the THM rate. I examine the increase in nuclear energy production, increase in strength of convection, and change in element abundance within the star.



<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/Astrophysical_Sim_Research/README_images/sim_screenshot.PNG?raw=true" width="600" height="600">
</p>


Here, I plot the nuclear energy production right around the main surface of C12-C12 burning, showing increased nuclear generation at three separate timesteps due to the enhanced fusion rates.

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/Astrophysical_Sim_Research/README_images/nuc_gen.PNG?raw=true" width="1000" height="450">
</p>

This shows the abundance of the three isotopes involved in the enhanced fusion reaction, centered around the burning surface. We see less C12, as it is being burned and more Mg24 and Ne20, as they are being produced.

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/Astrophysical_Sim_Research/README_images/fusion_eq.PNG?raw=true" width="1000" height="450">
</p>

Showing the strength of convection throughout the simulation. The enhanced fusion reaction rates found using the THM causes an increase in energy generation, and thus convection in the star.

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/Astrophysical_Sim_Research/README_images/convection.PNG?raw=true" width="800" height="550">
</p>

In summary, we found an increase of nuclear energy production during a C12-C12 burning phase as a result of the enhanced fusion reaction rates. This undoubtedly will have an affect on a star's evolution, as the convection and abundance of several isotopes have been shown to be significantly effected in our 20,000s simulations. To fully understand what the affects would be, longer simulations would need to be conducted. This would likely have to occur in a simulation environment like MESA, as these 2-D simulations in FLASH were incredibly computationally expensive, taking hundreds of thousands of core-hours over the course of months. 






