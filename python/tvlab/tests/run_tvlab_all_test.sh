#!/bin/bash

# utils
python -m unittest test_utils_basic -v


# cv
chmod u+x ./run_tvlab_cv_test.sh
./run_tvlab_cv_test.sh


# defect_detector
python -m unittest test_defect_detector_basic_detector -v
python -m unittest test_defect_detector_phot_detector -v
python -m unittest test_defect_detector_match_template_detector -v

python test_defect_detector_autoencoder_detector.py
python test_defect_detector_mahalanobis_detector.py


# category
python -m unittest test_category_image_data -v
python test_category_fast_category.py
python test_category_category_experiment.py

python -m unittest test_category_multi_task_image_data -v
python test_category_multi_task_category.py

python -m unittest test_category_batch_sampler -v
python test_category_fast_similar_cnn.py


# detection
python -m unittest test_detection_pascal_voc_io -v
python -m unittest test_detection_bbox_overlaps -v

python -m unittest test_detection_image_data -v
python test_detection_fast_detection.py -v
python test_detection_fast_mmdetection.py
python test_detection_eval_detection.py -v


# instance_segmentation
python -m unittest test_segmentation_image_data -v
python -m unittest test_segmentation_polygon_overlaps
python test_segmentation_fast_segmentation.py
python test_segmentation_fast_mmsegmentation.py


# ocr
# python test_fast_ocr.py
