#!/bin/bash

# cv
python -m unittest test_cv_basic -v
python -m unittest test_cv_geometry -v
python -m unittest test_cv_caliper_interface -v
python -m unittest test_cv_perspective -v
python -m unittest test_cv_qr_decode -v

python test_cv_blob.py
python test_cv_blobtool.py
python test_cv_chessboard_calibration.py
python test_cv_color_checker.py
python test_cv_color_segmenter.py
python test_cv_shape_based_matching.py
python test_cv_template_based_matching.py
python test_cv_xld.py

# filter.py will conflict with python filter() module
# no need to test it.
# python -m unittest test_cv_filter -vs