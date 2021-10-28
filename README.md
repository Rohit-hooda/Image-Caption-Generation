## Image Captioning - Team SaaS
A repository to collaborate on the Research Project published in the GIT-JET journal 2021.

## Software Requirements
To work on the project, please use Python 3.
Install the required packages from [requirements.txt](./requirements.txt)
```bash
pip install --user -r requirements.txt
```
NOTE - These were the packages that were default present in the UCSD DSMLP cluster. They have been uploaded here so as to ensure compatibility for runs.


### Dataset Annotations (Captions) Download
These are the steps to set up the dataset:-
1. Just use the images of the dataset given in the DSMLP cluster in /datasets/COCO-2015/
2. Create a sub-directory in the project root named datasets/
3. Create another sub-directory within it named as COCO/
4. For the captions, download the annotations from the MS COCO website.
5. Download the training set annotations as a zip file from [here](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
6. Inflate inside the ./datasets/COCO/
7. Similary, download the zip file for 2014 testing image information from [here](http://images.cocodataset.org/annotations/image_info_test2014.zip) and follow step 6.
8. Similarly, download the zip file for the 2015 testing image information from [here](http://images.cocodataset.org/annotations/image_info_test2015.zip) and follow step 6.

### Image loader
These are the steps to get your COCO dataset image loader up and running :-

1. Clone this repo recursively.
To do this, run
```bash
cd src/
git submodule update --init --recursive
```
2. Build the submodule by running the following
```bash
cd src/cocoapi/PythonAPI/
make
```
3. Additionally symlink the pycocotools in the cocoapi's PythonAPI directory into src/
This can be done by the following
```bash
cd src/
ln -s ./cocoapi/PythonAPI/pycocotools ./
``` 
