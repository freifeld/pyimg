#!/usr/bin/env python
"""
Author: Oren Freifeld
Email: freifeld@dam.brown.edu
"""
import numpy as np
import cv2
from cv2 import cv
from of.utils import FilesDirs
#from pyvision.essentials import np,cv2
from _Img import Img

 

class Imgs(object):    
    def __init__(self):
        self.imgs = dict()
    def set_img(self,key , img):
        """This function tries to do things inplace"""
        if not isinstance(img,Img):
#            pass
            raise TypeError(type(img))
        try:
            self.imgs[key][:] = img;
        except KeyError:
            self.imgs[key] = img
            try:
                getattr(self,key)
            except:
                setattr(self,key,self.imgs[key])   
        except (NameError,ValueError) as e:
            self.imgs[key] = img
            try:
                getattr(self,key)
            except:
                setattr(self,key,self.imgs[key]) 
    def keys(self):
        return self.imgs.keys()                
    def get_shape(self,key):
        return self.imgs[key].shape       
    def get_shape_2d(self,key):
        return self.imgs[key].shape[:2]         
    def get_img(self,key):
        return self.imgs[key]
    def zeros(self,key,shape,dtype):
        """This function tries to do things inplace"""  
        try:
            img = self.imgs[key]                        
        except KeyError:
            self.set_img(key,Img(np.zeros(shape,dtype)))
            return
            
        if img.shape == shape and img.dtype == dtype:
            img.fill(0)  # TODO: is this the fastest way?
#                self.imgs[key].__setitem__(self.imgs[key]!=0,0) 
        else:           
            self.set_img(key,Img(np.zeros(shape,dtype)))
    def set_the_nonzeros_to_zero(self):  
        # TODO: is this the fastest way?
#       [img.__setitem__(img!=0,0) for img in self.imgs.itervalues()] 
        [img.fill(0) for img in self.imgs.itervalues()] 
 
                  
    def imread(self,key,filename_fullpath ,
               read_grayscale = False):
        """This function tries to do things inplace"""     
        FilesDirs.raise_if_file_does_not_exist(filename_fullpath)         
        try:
            self.imgs[key][:] = cv2.imread(filename_fullpath) 
#            print "YES!"                      
        except (KeyError,NameError,ValueError) as e:
            self.set_img(key,Img(filename_fullpath,read_grayscale = read_grayscale))
        

            
            
            
    def undistort(self,key,cam):
##        self.set_img(key, cv.undistort(self.imgs[key],
##                                       cam.np_camera_matrix,
##                                       cam.np_dist_coeffs))
        self.set_img(key, Img.undistort(self.imgs[key],cam ))
    def imresize(self,key,factor):
        return np.asarray(
                cvdo.imresize(cv.fromarray(self.imgs[key]),
                              factor ))
    def roipoly(self,key):
        mask = cvdo.UI.roipoly(win_name = key,img = self.imgs[key])
        return mask
    def imshow_all(self,ch=None,use_abs = False,divide_by_max=False):
        """
        ch means channel. If None, all channels are used.
        """
        f = [lambda x:x,lambda x:np.abs(x)][use_abs]
        
        if ch is None:
            [f(self.imgs[k]).imshow(k,divide_by_max=divide_by_max) for k in self.imgs] 
        else:
            # Force copy as o.w. opencv complains.             
            [f(self.imgs[k][:,:,ch].copy()).imshow(k,divide_by_max=divide_by_max) 
                    for k in self.imgs]             
    def imshow(self,win_name,key = None):             
        if key == None:
            key = win_name
        self.imgs[key].imshow(win_name)

    def distance_transform(self,key):
        """opencv expects uint8 and zeros values fro the mask.
            So negate and convert"""
##        dt = cv.distanceTransform(
##            np.uint8(False == self.imgs[key]),cv.CV_DIST_L2,5)
        dt = Img.distance_transform(img = self.imgs[key])
        return dt
    def signed_distance_transform(self,key,rect = None , sdt = None):
        
        if rect == None:
##            sdt = Img.signed_distance_transform(mask = self.imgs[key])
            mask = self.imgs[key]
            sdt = mask.signed_distance_transform()
        else:
            if sdt == None:
                sdt = Img(np.zeros(self.imgs[key].shape[0:2] , dtype  = np.float32) )         
##            sdt[rect.ymin:rect.ymax,rect.xmin:rect.xmax] =(Img.signed_distance_transform(
##                img = rect.get_subrect(img = self.imgs[key])))
            
##            sdt[rect.ymin:rect.ymax,rect.xmin:rect.xmax] =Img.signed_distance_transform(
##                mask = self.imgs[key][rect.ymin:rect.ymax,rect.xmin:rect.xmax],
##                )
#            Img.signed_distance_transform(
#                mask = self.imgs[key],
#                rect = rect , sdt = sdt)
            mask = self.imgs[key]
             
            mask.signed_distance_transform(rect = rect , sdt = sdt)  
        return sdt
