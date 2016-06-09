# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:01:14 2016

@author: freifeld
"""

import cv2
def _is_opencv_legacy():
    return cv2.__version__.startswith('2')
    
is_opencv_legacy = _is_opencv_legacy()
    
