# UW-Madison-GI-Tract-Image-Segmentation


Solution Ranking : Bronze Medal
----

<img src="https://www.google.com.tw/url?sa=i&url=https%3A%2F%2Fwww.kaggle.com%2Fcompetitions%2Fuw-madison-gi-tract-image-segmentation&psig=AOvVaw0_fqWMnwi_ZcCdQSzhX8Z5&ust=1668173328585000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCNCUnY3co_sCFQAAAAAdAAAAABAJ">

[Kaggle Competition Link : Click Here](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation)

Background 
----

Track healthy organs in medical scans to improve cancer treatment
In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X-ray beams pointed to tumors while avoiding the stomach and intestines. With newer technology such as integrated magnetic resonance imaging and linear accelerator systems, also known as MR-Linacs, oncologists are able to visualize the daily position of the tumor and intestines, which can vary day to day. In these scans, radiation oncologists must manually outline the position of the stomach and intestines in order to adjust the direction of the x-ray beams to increase the dose delivery to the tumor and avoid the stomach and intestines. This is a time-consuming and labor intensive process that can prolong treatments from 15 minutes a day to an hour a day, which can be difficult for patients to tolerate—unless deep learning could help automate the segmentation process. A method to segment the stomach and intestines would make treatments much faster and would allow more patients to get more effective treatment.

The UW-Madison Carbone Cancer Center is a pioneer in MR-Linac based radiotherapy, and has treated patients with MRI guided radiotherapy based on their daily anatomy since 2015. UW-Madison has generously agreed to support this project which provides anonymized MRIs of patients treated at the UW-Madison Carbone Cancer Center. The University of Wisconsin-Madison is a public land-grant research university in Madison, Wisconsin. The Wisconsin Idea is the university's pledge to the state, the nation, and the world that their endeavors will benefit all citizens.

In this competition, you’ll create a model to automatically segment the stomach and intestines on MRI scans. The MRI scans are from actual cancer patients who had 1-5 MRI scans on separate days during their radiation treatment. You'll base your algorithm on a dataset of these scans to come up with creative deep learning solutions that will help cancer patients get better care.

In this figure, the tumor (pink thick line) is close to the stomach (red thick line). High doses of radiation are directed to the tumor while avoiding the stomach. The dose levels are represented by the rainbow of outlines, with higher doses represented by red and lower doses represented by green.

Cancer takes enough of a toll. If successful, you'll enable radiation oncologists to safely deliver higher doses of radiation to tumors while avoiding the stomach and intestines. This will make cancer patients' daily treatments faster and allow them to get more effective treatment with less side effects and better long-term cancer control.


Acknowledgments
----

Sangjune Laurence Lee MSE MD FRCPC DABR
Poonam Yadav Ph.D., DABR
Yin Li PhD
Jason J. Meudt BS, RTT
Jessica Strang
Dustin Hebel
Alyx Alfson MS CMD, R.T.(T)
Stephanie J. Olson RTT (BS), CMD (MS)
Tera R. Kruser MS, RTT, CMD
Jennifer B Smilowitz, Ph.D., DABR, FAAPM
Kailee Borchert
Brianne Loritz
John Bayouth PhD
Michael Bassetti MD PhD

Work funded by the University of Wisconsin Carbone Cancer Center Pancreas Pilot Research Grant.


Dataset Description
----

In this competition we are segmenting organs cells in images. The training annotations are provided as RLE-encoded masks, and the images are in 16-bit grayscale PNG format.

Each case in this competition is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal of this competition is to be able to generalize to both partially and wholly unseen cases.

Note that, in this case, the test set is entirely unseen. It is roughly 50 cases, with a varying number of days and slices, as seen in the training set.

How does an entirely hidden test set work?
The test set in this competition is only available when your code is submitted. The sample_submission.csv provided in the public set is an empty placeholder that shows the required submission format; you should perform your modeling, cross-validation, etc., using the training set, and write code to process a non-empty sample submission. It will contain rows with id, class and predicted columns as described in the Evaluation page.

When you submit your notebook, your code will be run against the non-hidden test set, which has the same folder format (<case>/<case_day>/<scans>) as the training data.

<strong>Files</strong>

- train.csv - IDs and masks for all training objects.
- sample_submission.csv - a sample submission file in the correct format

train - a folder of case/day folders, each containing slice images for a particular case on a given day.
Note that the image filenames include 4 numbers (ex. 276_276_1.63_1.63.png). These four numbers are slice height / width (integers in pixels) and heigh/width pixel spacing (floating points in mm). The first two defines the resolution of the slide. The last two record the physical size of each pixel.

Physical pixel thickness in superior-inferior direction is 3mm.

<strong>Columns</strong>
- id - unique identifier for object
- class - the predicted class for the object
- EncodedPixels - RLE-encoded pixels for the identified object

