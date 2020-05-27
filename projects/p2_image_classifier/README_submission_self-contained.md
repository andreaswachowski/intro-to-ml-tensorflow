This zip file contains the submission to Udacity's "Create your own Image
classifier" project, as part of the "Introduction to Machine Learning"
Nanodegree.

# Contents

The primary, self-developed files are

    * Project_Image_Classifier_Project.ipynb
    * predict.py

with associated files

    * Project_Image_Classifier_Project.html
    * predict_utils.py
    * requirements.txt

For convenience, the submission also includes the generated model

    * flower_model.h5

and some files originally provided by Udacity:

    * test_images/*
    * label_map.json

# Environment setup

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Development was done initially on a Macbook using Python 3.7.7,
then switched to Linux with a CUDA-capable graphics card and
Python 3.8.3.
