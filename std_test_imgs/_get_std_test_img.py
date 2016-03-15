#!/usr/bin/env python
"""
Created on Fri Jan 31 10:05:11 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import os
import inspect

def get_std_test_img(res=None,num=1,grey=True):
    dirname_of_this_file = os.path.dirname(inspect.getfile(inspect.currentframe()))

    if res is not None:
        raise NotImplementedError
    
    if not grey:
        raise NotImplementedError  
    filenames=[os.path.join(dirname_of_this_file,'g512_001/1.pgm'), # Barbara
               os.path.join(dirname_of_this_file,'g512_001/11.pgm'),
               os.path.join(dirname_of_this_file,'g512_001/13.pgm'),
               os.path.join(dirname_of_this_file,'g512_001/47.pgm'),
               os.path.join(dirname_of_this_file,'g512_001/48.pgm')]
    return filenames[num]
    
    
if __name__ == "__main__":
    from pyimg import *
    img = Img(get_std_test_img())
    
    cv2destroyAllWindows()
    img.imshow()
