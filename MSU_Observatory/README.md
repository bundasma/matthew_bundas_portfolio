## Operating Michigan State University's Campus Observatory

From March 2017 to August 2019, the time of my graduation from MSU, I worked as and Undergraduate Observer, where I was responsible for the end-to-end operation of the observatory. This gave me very valuable experience working hands on with a telescope and working with real-world data used by astronomers. Given the Michigan weather and proximity to Michigan's capital, we often did not have the best observing conditions and focused on variable star targets as observing the change in brightness of stars was something we were very capable of doing. On a typical night, I would open the observatory, take calibration images (bias, darks, flats), and take dozens of long-exposure images of a handful of targets throughout the night. Afterwards, I would reduce the data by applying the calibration frames, stacking the images, and taking photometry measurements. Once the data was reduced, I would distribute the data to various organzations such as the American Associtation of Variable Star Observers and the Center for Backyard Astronomy. After a year of so of observing, I was promoted to an "expert observer", a leadership role where I gained more responsibilities and helped train my peers. The MSU Observatory also held public observing nights on a monthly basis, which I volunteered for as much as I could. 


## Calibration Frames
Every observing night, before taking science images, calibration frames were taken to help with data reduction. These frames, when applied to science images, help to remove systematic effects associated with the camera, ideally leaving behind just light counts directly as a result of stars. 


### Bias
Bias frames were obtained by taking images with the shortest exposure time possible and the aperature closed. This helped capture the baseline pixel values, which could be subtracted off of the science images to obtain an image that better represents the light coming from stars. 

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/MSU_Observatory/README_images/bias.png?raw=true" width="450" height="450">
</p>


### Darks
Darks were obtained by taking an image with the same exposure length as the science images and the aperature closed. Darks help measure the sort of random noise pixels might have, where they register photons of light, despite no light reaching the camera. Similar to Bias frames, Darks were subtracted off of science images to remove random noise associated with individual pixels.

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/MSU_Observatory/README_images/dark.png?raw=true" width="450" height="450">
</p>


### Flats
Flats were taken just after sunset, where the camera was exposed to enough uniform light to be just under saturation values. Because the light in reality is uniform, differing values in pixel counts revealed differing rates pixels registered counts due to error in pixels or light filter used. These frames could be normalized to have a mean value of ~1 and then the science images could be divided by the flat images to remove systematic counting bias.

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/MSU_Observatory/README_images/flat.png?raw=true" width="600" height="450">
</p>


## Science Images 
Science images for the MSU Observatory were typically 60-180 second exposures of which we'd try to take as many as possible over the course of the night. In most cases, variable stars were our targets and required data spanning as many periods as possible. We would find our targets by instructing the telescope to point at the provided coordinates, and zero in on them using star charts to help make intelligent adjustments. 

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/MSU_Observatory/README_images/starfield.png?raw=true" width="450" height="450">
</p>



## Data Reduction
Data reduction required following several steps involving AstroImageJ and python to ultimately produce a text file showing the magnitude change of a variable star over the course of the observing period. First, the calibration frames were applied to the science images to remove as much systematic bias as possible. Following calibration, photometry was performed by stacking the images so that stars within them were located in the same positions and then obtaining magnitude measurements using comparison stars of known magnitude. Finally, python scripts were used to graph and validate the data and organize the data according to specifications provided by AAVSO and others. 


<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/MSU_Observatory/README_images/reduction.png?raw=true" width="450" height="450">
</p>

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/MSU_Observatory/README_images/measurements.png?raw=true" width="650" height="450">
</p>

