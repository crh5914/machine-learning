import cv2
import numpy as np
def getOriginLBPFeature(im):
	h,w = im.shape
	des = np.zeros((h-1,w-1))
	for i in range(1,h-1):
		for j in range(1,w-1):
			des[i-1][j-1] += (im[i-1][j-1]>im[i][j])
			des[i-1][j-1] += (im[i-1][j]>im[i][j])<<1
			des[i-1][j-1] += (im[i-1][j+1]>im[i][j])<<2
			des[i-1][j-1] += (im[i][j+1]>im[i][j])<<3
			des[i-1][j-1] += (im[i+1][j+1]>im[i][j])<<4
			des[i-1][j-1] += (im[i+1][j]>im[i][j])<<5
			des[i-1][j-1] += (im[i+1][j-1]>im[i][j])<<6
			des[i-1][j-1] += (im[i][j-1]>im[i][j])<<7
	return des
im = cv2.imread('001.png')
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow('origin',im)
des = getOriginLBPFeature(im)
print(des)
cv2.imshow('lbp',des)
cv2.waitKey(0)