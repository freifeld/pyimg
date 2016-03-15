import numpy as np
import cv2

_imshow = cv2.imshow

def imshow(self,winname,
                     option,float_min_val,float_max_val,divide_by_max = False):
     
    if divide_by_max == True:
        float_max_val = self.max()
#    pyvision.ipshell()
    if not self.dtype in (np.float32,np.float64,np.uint16):#'float32':        
        try:       # Start optimistic... After all, "Hope springs ethernal."
            _imshow(winname,self)
            return
        except:            
            # if numpy played with the memory, opencv won't like it.
            # copy() should solve it. 
            if self.dtype == bool: # bool is not supported in _imshow        
                _imshow(winname,255 * self.astype(np.uint8).copy())
##                _imshow(winname,255 * np.uint8(self).copy())
                return
            _imshow(winname,self.copy())
            return
    elif self.dtype == np.uint16:
        _imshow( winname,self.astype(np.float)/self.max())
        
    else:  # oh, well... 
        # If min/max is 0/1 then just let opencv do its thing:
        # map [0,1] to [0,255]
        if float_min_val == 0.0 and float_max_val == 1.0:
            _imshow(winname,self)  # the easy case.
        else:
            if float_max_val <= float_min_val:
                raise ValueError
            if float_min_val >= 0:
                #We will show a grayscale image.                  
                # We are going to modify the image, so create a
                # copy to avoid affecting the source.
                _self  =  self.copy() 
                _self -=  float_min_val
                _self /=  (float_max_val - float_min_val)
                # should now be btwn 0 and 1
                _imshow(winname,_self)
            else:
                # We will show an RGB image.
                if self.ndim != 2:
                    raise ShapeError(self.shape)
                _self_n = self.copy()
                _self_p = self.copy()
                 
                          
                _self_p[(_self_p<0).nonzero()]=0.0  # is this the fastest way? 
                _self_n[(_self_n>0).nonzero()]=0.0
                _self_z = np.zeros_like(self)
                _self_z[(self==0).nonzero()] = 1.0
                R = _self_p / float_max_val
                G = _self_z
                B = _self_n / (float_min_val) # B should now be positive.
                RGB_imsz = list(self.shape)+[3]
                _self = np.zeros( RGB_imsz,dtype = self.dtype)
                _self[:,:,0] = R
                _self[:,:,1] = G
                _self[:,:,2] = B
                _imshow(winname,_self)
