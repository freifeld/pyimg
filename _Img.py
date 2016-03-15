#!/usr/bin/env python
"""
This is one of my first python files ever... And it shows. 

It was created with an old version of opencv, before the introduction 
of "cv2".
Whenever I have the time I update it incrementally by replacing
the cv stuff with the cv2 alternatives.

Author: Oren Freifeld
Email: freifeld@dam.brown.edu
"""

import numpy as np
import cv2
from cv2 import cv
from of.utils import FilesDirs
#from pyvision.essentials import *
#from pyvision.core.cvdo import Rect
from _imshow import imshow as _imshow


_NamedWindow = cv2.namedWindow


_lcviplimage = cv.iplimage
_lcvGetMat = cv.GetMat
_lcvGetImage = cv.GetImage
_lcvCreateMat = cv.CreateMat
_lcvmat = cv.cvmat
_lcvfromarray = cv.fromarray

_lasarray = np.asarray

_lzeros = np.zeros
_lzeros_like = np.zeros_like
_lempty  = np.empty
_lempty_like = np.empty_like

class Img(np.ndarray):
    """An image class. Derived from np.ndarray.
       In addition to std constructs, can also be initialized by Img(filename)
       Version 0.0"""
    def __new__(cls, input_array, info=None , read_grayscale = False):
        #----- enabling construction from a filename -------------
        if type(input_array) in [str,np.string_]:  # Assume it is a filename.  
            input_array = cls.imread(input_array , read_grayscale = read_grayscale)
        
        #----- enabling construction from an cv.iplimage----------
        elif type(input_array) == _lcviplimage:
            input_array = cls.ipl2np(input_array)

        #----- enabling construction from an cv.cvmat-------------
        elif type(input_array) == _lcvmat:
            input_array = cls.cvmat2np(input_array)


        # Verify we have a numpy ndarray.    
        if not isinstance(input_array,np.ndarray):
            print "Expected isinstance(input_array,np.ndarray) == True"
            raise TypeError(type(input_array))

        # Now follow the example in python's online doc.
        
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = _lasarray(input_array).view(cls)
       # obj = _lasarray(input_array).view(cls).astype(input_array.dtype)

        # TODO: I don't really use info at the moment. I keep it as a reminder.
        
        # add the new attribute to the created instance
        obj.info = info

        obj.rows = obj.shape[0]
        obj.cols = obj.shape[1]
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # This is important - otherwise only the explicit construction would work.
        # See PEP.
        
        if obj is None: return # in this case, we don't need to return.

        # TODO: again, I keep info as a reimnder.
        self.info = getattr(obj, 'info', None)
        
    def imshow_pylab(self):
        plt.imshow( self ,interpolation = 'nearest' )
    def imshow(self,winname = None ,
               option = 'opencv',
               float_min_val = 0.0,
               float_max_val = 1.0,
               divide_by_max=False):        
        """A thin wrapper to imshow"""
        # With time, the wrapper became a bit too long...
        # So it has its own module now.    
        if not option == 'opencv':
            print """TODO: add the pylab option"""
            raise NotImplementedError        
        if winname is None:
            winname = 'winname'
        if not isinstance(winname,str):
            raise TypeError(type(winname))
        _NamedWindow(winname,cv.CV_WINDOW_NORMAL)


        _imshow(self,winname,
                     option,float_min_val,float_max_val,divide_by_max)
##
##        if not self.dtype == 'float32':        
##            try:       # Start optimistic... After all, "Hope springs ethernal."
##                cv.imshow(winname,self)
##            except:            
##                # if numpy played with the memory, opencv won't like it.
##                # copy() should solve it. 
##                if self.dtype == bool: # bool is not supported in cv.imshow        
##                    cv.imshow(winname,255 * self.astype(np.uint8).copy())
##    ##                cv.imshow(winname,255 * np.uint8(self).copy())
##                    return
##                cv.imshow(winname,self.copy())
##        else:
##            # If min/max is 0/1 then just let opencv do its thing:
##            # map [0,1] to [0,255]
##            if float_min_val == 0.0 and float_max_val == 1.0:
##                cv.imshow(winname,self)  # the easy case.
##            else:
##                if float_max_val <= float_min_val:  raise ValueError
##                if float_min_val >= 0:
##                    #We will show a grayscale image.                  
##                    # We are going to modify the image, so create a
##                    # copy to avoid affecting the source.
##                    _self  =  self.copy() 
##                    _self -=  float_min_val
##                    _self /=  (float_max_val - float_min_val)
##                    # should now be btwn 0 and 1
##                    cv.imshow(winname,_self)
##                else:
##                    # We will show an RGB image.
##                    if self.ndim != 2:
##                        raise ShapeError(self.shape)
##                    _self_n = self.copy()
##                    _self_p = self.copy()
##                     
##                              
##                    _self_p[(_self_p<0).nonzero()]=0.0  # is this the fastest way? 
##                    _self_n[(_self_n>0).nonzero()]=0.0
##                    _self_z = _lzeros_like(self)
##                    _self_z[(self==0).nonzero()] = 1.0
##                    R = _self_p / float_max_val
##                    G = _self_z
##                    B = _self_n / (float_min_val) # B should now be positive.
##                    RGB_imsz = list(self.shape)+[3]
##                    _self = _lzeros( RGB_imsz,dtype = self.dtype)
##                    _self[:,:,0] = R
##                    _self[:,:,1] = G
##                    _self[:,:,2] = B
##                    cv.imshow(winname,_self)

        return self
    @staticmethod
    def waitKey(delay = None):
        if delay == None:
            return cv2.waitKey()
        else:
            return cv2.waitKey(delay)
    @staticmethod
    def WaitKey(delay = None):
        if delay == None:
            return cv.WaitKey()
        else:
            return cv.WaitKey(delay)


    def toggle_landscape_vs_portrait(self ):
        print """TODO: need first to check if L of P, and then treat accordingly"""
        raise NotImplementedError
        if self.ndim == 3:
            axes = [1,0,2]
            return np.ndarray.transpose(self,*axes)[:, ::-1,:]
        elif self.ndim == 2:
            return np.ndarray.transpose(self)[ ::-1,:]
        else:
            raise ShapeError(self.shape)

    def imwrite(self,filename, override = False,create_dir_if_needed = False):
        if 'raw' in filename.lower():
            raise RawDataError(filename)
        if not override:
            FilesDirs.raise_if_file_already_exists(filename)
        try:
            dirname = os.path.dirname(filename)
            FilesDirs.raise_if_dir_does_not_exist(dirname)
        except DirDoesNotExistError:
            if create_dir_if_needed:
                FilesDirs.mkdirs_if_needed(dirname)                
            else:
                raise
        if self.dtype == bool:
            ret = cv2.imwrite(filename , 255*self.astype(np.uint8))            
        
        else:
##            from of.utils import ipshell
##            ipshell('s')
##            2/0
            ret = cv2.imwrite(filename , self )
        if not ret:                       
            raise CvImwriteError(filename)

    def query_frame_inplace(self,capture):
#        # Convert to cvmat is enough.
#        img_ipl = cv.QueryFrame(capture)
#        if img_ipl != None:  # i.e. still cpaturing. 
#            self[:] = _lcvGetMat( img_ipl )
#        else:
#            return 1
        raise NotImplementedError("PLease swtich from cv to cv2")

    def imread_inplace(  self, filename_fullpath , read_grayscale = False):
        FilesDirs.raise_if_file_does_not_exist(filename_fullpath)
        try:
            if not read_grayscale:
                self[:] =  cv2.imread(filename_fullpath)
            else:
                self[:] = cv2.imread(filename_fullpath,cv.CV_LOAD_IMAGE_GRAYSCALE)
#            print "Look! It's working!"
        except:
            raise
#            raise CvImreadError(filename_fullpath )
            
            raise CvImreadError(filename_fullpath)
    @staticmethod
    def imread(filename_fullpath , option = 'opencv',read_grayscale = False):
        """A thin wrapper to imread. This is a static method."""
        FilesDirs.raise_if_file_does_not_exist(filename_fullpath)
        if not option == 'opencv':
            print """TODO: add the pylab option"""
            raise NotImplementedError
 
        try:
            if not read_grayscale:
                return cv2.imread(filename_fullpath)
            else:
                return cv2.imread(filename_fullpath,cv.CV_LOAD_IMAGE_GRAYSCALE)
        except:
            raise CvImreadError(filename_fullpath )


    @staticmethod
    def ipl2np(img_ipl):
        img_cvmat = _lcvGetMat(img_ipl)    # ipl2cvmat
        return Img.cvmat2np(img_cvmat)        
    @staticmethod
    def cvmat2np(img_cvmat):
        return _lasarray(img_cvmat)
    @staticmethod
    def cvmat2ipl(img_cvmat):
        return _lcvGetImage(img_cvmat)
    @staticmethod
    def img2cv(img):
        return _lcvfromarray(img.copy())
    @staticmethod
    def img2ipl(img):
        return Img.cvmat2ipl(Img.img2cv(img))
    @staticmethod
    def get_bb(mask):
        if mask.ndim != 2:  raise NotImplementedError           
        nz = mask.nonzero()
        if len(nz[0])==0 or len(nz[1])==0:
            raise ValueError        
#        xmin = min(nz[1])
#        ymin = min(nz[0])
#        xmax = max(nz[1])
#        ymax = max(nz[0])
        xmin = nz[1].min()
        ymin = nz[0].min()
        xmax = nz[1].max()
        ymax = nz[0].max()
        rect = Rect(xmin,ymin,xmax-xmin+1,ymax-ymin+1,thickness = 1)
        return rect 
    
    def distance_transform(self,dt = None):
        """self is the bw. opencv expects uint8 and zeros values for the mask.
            So negate by 255-self. Note: assumes nonzeros are 255."""
        if self.dtype != np.uint8:
            raise TypeError(self.dtype)
        if dt == None:
#            raise Warning('It is (much) better to pass dst as an arguemnet')
#            ipshell('hi')
            dt = cv2.distanceTransform((False == self).astype(np.uint8),
                                              cv.CV_DIST_L2,5)
            return Img(dt)
        else:
            if dt.dtype != np.float32:
                raise TypeError(dt.dtype)
##            cv2.distanceTransform((False == self).astype(np.uint8),
##                                  cv.CV_DIST_L2,5,dst = dt)
            
            #cpp and numpy. 
            #cv2.distanceTransform(255-self,cv.CV_DIST_L2,5,doer.dt1)

            # c. More than twices fast. Maybe it's just the flags? Need to check.
            cv.DistTransform(_lcvfromarray(255-self),_lcvfromarray(dt),cv.CV_DIST_L2,mask_size=3)


    def signed_distance_transform(self ,rect = None,
                                  sdt = None ,
                                  two_dt_bufs = None):
        """self is the bw"""
        if sdt != None and not isinstance(sdt,Img):
            raise TypeError(type(sdt))
        if two_dt_bufs == None:
            two_dt_bufs = ( _lempty(self.shape,np.float32),
                            _lempty(self.shape,np.float32))
             
 
        if rect == None:
            self.distance_transform(  two_dt_bufs[0] )
            (255 - self).distance_transform(  two_dt_bufs[1])
            if sdt == None:
                sdt = two_dt_bufs[0] - two_dt_bufs[1]
                return sdt
            else: # inplace
                sdt[:] = two_dt_bufs[0] - two_dt_bufs[1]
                return
        else:
            r = rect
            if sdt == None:
                raise NotImplementedError
            else: # inplace
                1/0
                sdt[r.ymin:r.ymax,r.xmin:r.xmax] = (  
                                                (self[r.ymin:r.ymax,
                                                      r.xmin:r.xmax]).distance_transform() -
                                                (False == self[r.ymin:r.ymax,
                                                      r.xmin:r.xmax]).distance_transform())
                    

             
                return

         
    def get_diagonal_length(self):
        return norm(self.shape[0:2])

    
    def imshow_signed_distance_transform(self,winname):
        """self is the sdt"""
        sdt = self
        _sdt = sdt.copy()
        _sdt[(np.absolute(_sdt)<=1).nonzero()] = 0
        _sdt.imshow(winname = winname,
                   float_min_val = -_sdt.get_diagonal_length() / 5,
                   float_max_val =  _sdt.get_diagonal_length() / 5 )

    def imresize(self,factor  , verbose = False):
        if factor != 1.0:
            sz = (int(float(self.shape[0]) * factor),
                  int(float(self.shape[1]) * factor))
            if verbose:
                print 'factor = {0}. sz = {1}'.format(factor,sz)
#            input_cvmat  = _lcvfromarray(self)        
#            output_cvmat = _lcvCreateMat(sz[0],sz[1] ,  input_cvmat.type)
#            cv.Resize(input_cvmat,output_cvmat)    
#            Img(output_cvmat)
            
#            out = Img( cv2.resize(self,(sz[0],sz[1])))
            # I swear it used to be (height,width)...
            # But now it seems to be (width,height)
            out = Img( cv2.resize(self,(sz[1],sz[0])))
            return out
        else:
            return self.copy()
    def pyrdown(self,dst = None):
        
        if self.shape[0] % 2 !=0 or self.shape[1] % 2 !=0:
            raise ShapeError(self.shape)
##        sz_new = (self.shape[1]/2,self.shape[0]/2)
##            img_cvmat = _lcvfromarray(self.copy())
##            img_ipl = Img.cvmat2ipl(_lcvfromarray(img_cvmat))
##            out = cv.CreateImage(sz_new,img_ipl.depth, img_ipl.nChannels)
##
##            cv.PyrDown(img_ipl, out)
##            return Img(out)
        shape_new = [s/2 for s in self.shape[0:2]]
        if self.ndim == 3: shape_new.append(self.shape[2])
        if dst == None:                                  
            dst = Img(_lempty(shape_new,self.dtype))
            cv.pyrDown(self,dst)
            return dst
        else:
            if not isinstance(dst,Img):
                raise TypeError(type(dst))
            cv.pyrDown(self,dst)
        
                
    def bgr2hsv(self):
         """Assumes BGR ordering"""
         return Img(cv.cvtColor(self,cv.CV_BGR2HSV))
    


if __name__ == '__main__':
    cv2.destroyAllWindows()

    img = Img(np.random.rand(100,100,3))
   
    img.imshow('image')
