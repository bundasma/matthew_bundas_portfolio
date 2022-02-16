## GLOBE Observer Landcover Content Classification

The GLOBE Observer program is a NASA-sponsored citizen science project helping to understand the state and evolution of Earth's landcover content. Data is collected by ordinary people via a mobile application, where users take pictures of their surroundings and have the **option** to provide descriptions of those images, helping to build landcover content maps for scientists to use. My work's goal was to find a machine learning solution to provide labels to un-described images to increase the usefullness of the GLOBE dataset. The dataset however has many issues posing challenges to training a machine learning model, including class imbalance, label noise and poor image quality. I developed a novel framework to help overcome these issues, obtaining a machine learning model which outperforms manual citizen labeling. 

For a detailed view of the project, I reccommend checking out the slides for the presentation I performed for my Master's Final Exam.

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/NASA_GLOBE_Observer_Research/README_images/class_instances.PNG?raw=true" width="600" height="450">
</p>

### Dataset Challenges

**Missing Labels** - Just about 40% of images in the GLOBE Dataset have labels. In general, more data is a desirable feature of a dataset.

**Noisy Labels** - Overall citizen label accuracy is about 57%, with some classes' accuracy as bad as ~15%. Training off of unreliable data is a challenge.

**Class Imbalance** - Image content reflected what sort of environments people live near. The largest share of images is Urban at ~40%, with other classes, such as Wetland as low as ~1%. Imbalanced data tends to lead to a model biases towards common classes, with poor performance on rare classes.

**Poor Images** - Roughly 2% of images are very bad quality, without landcover content such as pictures of the sky, directly pointed at the ground or covered by thumbs.

### Semi-Supervised Active Learning Framework
The Semi-Supverised Active Learning Framework tries to tackle the dataset challenges by solving each problem individually and iteratively increasing the dataset size. We start with a small, manually labeled dataset we can trust of ~500 images and grow the training set using Semi-Supervised Learning and Active Learning. This allows us to circumvent the noisy labels, training only with images and labels we have more confidence in.

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/NASA_GLOBE_Observer_Research/README_images/SSAL_framework.png?raw=true" width="600" height="450">
</p>

### Key Concepts

**Manually Labeled Subset** - To avoid training off of noisy labels provided by citizens, we manually label a subset of images, providing us with a small, clean dataset we can build off of. Images which don't recieve manually-derived labels make up a pool of images we can intelligently label with our framework to increase the dataset size.

**Transfer Learning** - A major field of research is creating the best CNN to classify the ImageNet dataset, made up of millions of images. These models have learned the features useful in general image classification and can be fine-tuned for a specific domain. In our case, we used Google's Inception-V3 and adapted it to classify our image's content using fully connected layers on the end of the network.

**Semi-Supverised Learning** - In general, semi-supervised learning is training a model with a dataset which is made up of a labeled portion and an unlabeled portion, which the  GLOBE dataset certain lines up with. We used a method called pseudolabeling, which involves having the model make predictions on unlabeled images, and turning some of the most confident predictions into labels, increasing the number of images which have labels, growing our training set size.

**Prediction Confidence Boosting With Citizen Labels** - To boost the number of pseudolabels we can create during Semi-Supervised learning, we pull in the noisy citizen labels. With the formula shown below, we can combine our model's predictions and citizen predictions to create a more confident prediction, allowing more predictions to meet a confidence threshold to be used as a pseudolabel. The idea is that while the noisy citizen labels can't directly be trusted, they can still provide some supplemental information and help us out.

**Active Learning** - Another way to bolster a training dataset is to have more humans provide labels. Active learning helps to be more efficient with human labor by guiding which images would be most useful to have labeled by humans. Most of active learning involves having humans label images which the model currently understands the least, derived from performing calculations on the model's predictions for given images. We used a modified entropy calculation as our selection criteria, weighting images we suspect to belong to a rare class higher than images we expect to belong to a common class. This helps us find more instances of rare classes, allowing our model to have more examples to learn from.

**Iteration** - Starting with the small, manually labeled subset, we perform a set number (~10) iterations of the framework where in each iteration, we train the model on the currently available training set, make predictions on unlabeled images and perform semi-supervised and active learning to grow our training set, using the final model to make predictions on the test set.

### Results

Overall, with the Semi-Supervised Active Learning Framework, we were able to outperform citizen classifications by a notable 10% margin in accuracy. Our model achieved an accuracy of about 70%, while citizens have an accuracy of about 57%. The framework was able to grow the training set size from about 500 images to over 17,000 images through semi-supervised and active learning. Our boosting of predictions with citizen labels allowed us to create close to twice as many pseudolabels compared to the baseline. The modified, weighted entropy calculation found more instances of rare classes than the baseline, unmodified entropy calculation. 

### Training Set Size Increase
Shows the size and makeup of the training set size vs the framework iteration. The modified (citizen boosted) SSAL Framework has a much larger training set size.

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/NASA_GLOBE_Observer_Research/README_images/train_set_growth.PNG?raw=true" width="600" height="450">
</p>

### Accuracy
Shows raw accuracy of model classification by class. Compares results of if we directly trained on the citizen labels (Naive Baseline), SSAL with no boosting and regular entropy (Baseline SSAL), SSAL with boosting and modified entropy (Modified SSAL) as well as the citizen accuracy.

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/NASA_GLOBE_Observer_Research/README_images/GLOBE_final_accs.png?raw=true" width="600" height="450">
</p>

### Accuracy vs Number of Manually Labeled Images
Quantifies increase in accuracy as more images are manually labeled, comparing the modified and baseline SSAL experiments. This helps capture the efficiency of human labor of the modified SSAL, where for a given number of manually labeled images, the modified SSAL framework outperforms the baseline. These findings are largely a result of increased training set size from boosting the semi-supervised learning with citizen labels.

<p align="center">
<img src="https://github.com/bundasma/matthew_bundas_portfolio/blob/main/NASA_GLOBE_Observer_Research/README_images/GLOBE_SSAL.png?raw=true" width="600" height="450">
</p>



### Discussion

In performing Semi-Supervised Active Learning with the GLOBE dataset, we were able to create a model which outperforms citizen classification (57%) and outperforms the naive baseline of training directly on noisy labels (59%). We also showed that the modified SSAL outperforms the baseline, being more efficient with human labor and growing a larger training dataset. However, the model and framework has room for improvement, achieving an accuracy of 70%. Ideally, we would want the accuracy to be closer to 80-85%. I believe the deficieny in performance is likely for a few reasons. First, the SSAL Framework still does not fully overcome the issues with the dataset. Although we find more instances of rare classes with active learning, there is still a large class imbalance. Additionally, pseudolabels are not 100% accurate, so there still exists some label noise which negatively impacts training. It's also the nature of this problem that classification is difficult. Many classes look similar to eachother, ie Wetland and Open Water, and many images contain multiple classes, however the model has to choose just one class. More work could be performed to head towards a more balanced dataset, perhaps by running the framework for more iterations and then undersampling common classes and oversampling rare classes. This would help reduce the bias towards common classes like urban, which the model performs very well on compared to rare classes like Wetland where it really struggles. Pseuolabeling perofrmance could also be improved by experimenting more with the citizen-label boosting mechanism. I suspect there is some affect of using the citizen's accuracy to boost the model's predictions, as accuracy as a metric is inherently flawed, where high accuracy can be achieved simply by overpredicting. A more meaningful metric to evaluate citizen accuracy might be an F1 score which balances overpreprevision and recall. I would also be interestd in exploroing assigning multiple labels to an image rather than forcing a single label on a single image, as many images are more nuanced than the label suggests. This could be done with semantic-segmentation and , or another method of assessing percentage-wise composition of classes in an image. 















