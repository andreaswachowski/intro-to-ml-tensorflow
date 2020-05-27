#!/usr/bin/env bash

# Bundle files required for submission

mv requirements_minimal.txt requirements.txt

zip submission.zip \
	Project_Image_Classifier_Project.html \
	predict.py \
	predict_utils.py \
        requirements.txt

mv requirements.txt requirements_minimal.txt
