import cv2

from  is_opencv_legacy import  is_opencv_legacy
    
if is_opencv_legacy:
    from cv2 import cv


def cv2destroyAllWindows():
    try:
        cv2.destroyAllWindows() 
    except: 
        raise
#        print """
#        If you see an error about cvDestroyAllWindows,
#        It is safe to ignore it (it is realted to opencv2.3-->opencv2.4)
#        """
        
from _Img import Img
from _Imgs import Imgs
from std_test_imgs import get_std_test_img 



