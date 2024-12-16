#!/bin/bash

# No order
python -m unittest test_category_image_data -v
python -m unittest test_detection_image_data -v
python -m unittest test_segmentation_image_data.py -v

# Have order
python test_category_fast_category.py
python test_category_category_experiment.py

python test_detection_fast_detection.py
python test_detection_fast_mmdetection.py

python test_segmentation_fast_segmentation.py
python test_segmentation_fast_mmsegmentation.py