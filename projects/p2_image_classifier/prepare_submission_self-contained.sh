#!/usr/bin/env bash

# Bundle files required for submission

mv README.md README_repo.md
mv README_submission_self-contained.md README.md
mv requirements_self-contained.txt requirements.txt

zip submission.zip \
	Project_Image_Classifier_Project.html \
	Project_Image_Classifier_Project.ipynb \
	predict.py \
	predict_utils.py \
	label_map.json \
	flower_model.h5 \
	README.md \
        requirements.txt \
	assets/* \
	test_images/*

mv requirements.txt requirements_self-contained.txt
mv README.md README_submission_self-contained.md 
mv README_repo.md README.md
