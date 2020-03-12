#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import scipy
import scipy.stats as st
import matplotlib.pyplot as plt
import skimage.transform
import math
import sklearn.cluster

def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	filters = []
	scales = [5, 11]
	kernels = generate_kernel(scales)
	for kernel in kernels:
		# border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
		conFilter = cv2.Sobel(kernel, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_CONSTANT)
		# conFilter = cv2.Sobel(kernel, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
		orients = np.linspace(0, 360, 16)
		for i, orient in enumerate(orients):
			# imageFil=skimage.transform.rotate(conFilter, orient)
			# filters.append(imageFil)
			# imageFil=0
			rows, cols = conFilter.shape
			M = cv2.getRotationMatrix2D((cols / 2, rows / 2), orient, 1)
			dst = cv2.warpAffine(conFilter, M, (cols, rows))
			filters.append(dst)

	gabolFilters = []
	gFilters = gabor(scales, 0, 2, 0, 1)
	for gFilter in gFilters:
		orients = np.linspace(0, 360, 16)
		for i, orient in enumerate(orients):
			# gFil = skimage.transform.rotate(gFilter, orient)
			# gabolFilters.append(gFil)
			# gFil = 0
			rows, cols = gFilter.shape
			M = cv2.getRotationMatrix2D((cols / 2, rows / 2), orient, 1)
			dst = cv2.warpAffine(gFilter, M, (cols, rows))
			gabolFilters.append(dst)

	LMFilters = LM_filter()

	plot_DoG(filters, "DoG.png")
	plot_DoG(gabolFilters, "Gabor.png")
	plot_DoG(LMFilters, "LML.png", True)
	masks = generate_half_disks()
	for num in range(1,11):
		#Change hard coded path!!
		img = cv2.imread("/home/aruna/abaijal_hw0/Phase1/BSDS500/Images/" + str(num) + ".jpg",0)
		img_color = cv2.imread("/home/aruna/abaijal_hw0/Phase1/BSDS500/Images/" + str(num) + ".jpg",1)

		result1 = apply_filters(filters, img)
		result2 = apply_filters(gabolFilters, img)
		result3 = apply_filters(LMFilters, img)

		texton = combine_filters(img,result1,result2,result3, 'Texton'+str(num)+'.png')
		x,y = img.shape
		brightness = brightnessColor(img, x, y, 1, 'brightness'+str(num)+'.png')
		x,y,z = img_color.shape
		color = brightnessColor(img_color, x, y, z, 'color'+str(num)+'.png')
		texton_gradient = generate_gradient(texton, 64, masks)
		brightness_gradient = generate_gradient(brightness, 16, masks)
		color_gradient = generate_gradient(color, 16, masks)
		plt.imshow(texton_gradient)
		plt.savefig('texton_gradient'+str(num)+'.png')
		plt.close()

		plt.imshow(brightness_gradient)
		plt.savefig('brightness_gradient'+str(num)+'.png')
		plt.close()

		plt.imshow(color_gradient)
		plt.savefig('color_gradient'+str(num)+'.png')
		plt.close()
		# Change hard coded path!!
		sobelBaseline = cv2.imread(
			'/home/aruna/abaijal_hw0/Phase1/BSDS500/SobelBaseline/'+str(num)+'.png', 0)
		cannyBaseline = cv2.imread(
			'/home/aruna/abaijal_hw0/Phase1/BSDS500/CannyBaseline/'+str(num)+'.png', 0)

		pblite_out = np.multiply((texton_gradient+brightness_gradient+color_gradient)/3, (0.5 * cannyBaseline + 0.5 * sobelBaseline))

		plt.imshow(pblite_out,cmap='gray')
		plt.savefig('pblite'+str(num)+'.png')
		plt.close()

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""

def LM_filter():
	SUP = 49  # Support of the largest filter (must be odd)
	SCALEX = []
	for i in range(1, 4):
		SCALEX.append(np.sqrt(2) ** i)  # Sigma_{x} for the oriented filters
	NORIENT = 6  # Number of orientations

	NROTINV = 12
	NBAR = len(SCALEX) * NORIENT
	NEDGE = len(SCALEX) * NORIENT
	NF = NBAR + NEDGE + NROTINV
	F = np.zeros(shape=(SUP, SUP, NF))
	hsup = (SUP - 1) / 2
	[x, y] = np.meshgrid(np.arange(-hsup,hsup+1),np.arange(hsup,-hsup-1, -1))
	orgpts = [x.flatten(), y.flatten()]
	orgpts = np.array(orgpts)

	count = 1
	for scale in range(0, len(SCALEX)):
		for orient in range(0, NORIENT):
			angle = np.pi * orient / NORIENT
			c = math.cos(angle)
			s = math.sin(angle)
			rotpts = [[c, -s],[s, c]]
			rotpts = np.array(rotpts)
			rotpts = np.dot(rotpts, orgpts)
			F[:, :, count] = makefilter(SCALEX[scale], 0, 1, rotpts, SUP)
			F[:, :, count + NEDGE] = makefilter(SCALEX[scale], 0, 2, rotpts, SUP)
			count = count + 1

	count = NBAR + NEDGE
	SCALES = []
	for i in range(1, 5):
		SCALES.append(np.sqrt(2) ** i)
	for i in range(0, len(SCALES)):
		F[:, :, count] = gaussian2d(SUP, SCALES[i])
		F[:, :, count + 1] = log2d(SUP, SCALES[i])
		F[:, :, count + 2] = log2d(SUP, 3 * SCALES[i])
		count = count + 3
	return F

def makefilter(scale, phasex, phasey, pts, sup):
	gx = make_gauss_1d(3 * scale, 0, pts[0, :], phasex)
	gy = make_gauss_1d(scale, 0, pts[1, :], phasey)
	f = normalize(np.reshape(gx * gy, (sup, sup)))
	return f

def gaussian2d(sup, scales):
	var = scales * scales
	shape = (sup, sup)
	n, m = [(i - 1) / 2 for i in shape]
	x, y = np.ogrid[-m:m + 1, -n:n + 1]
	g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x * x + y * y) / (2 * var))
	return g

def log2d(sup, scales):
	var = scales * scales
	shape = (sup, sup)
	n, m = [(i - 1) / 2 for i in shape]
	x, y = np.ogrid[-m:m + 1, -n:n + 1]
	g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x * x + y * y) / (2 * var))
	h = g * ((x * x + y * y) - var) / (var ** 2)
	return h

def make_gauss_1d(sigma, mean, x, ord):
	num = []
	x = np.array(x)
	for i, val in enumerate(x):
		x[i] = val - mean
		num.append(x[i] * x[i])
	variance = sigma ** 2
	denom = 2 * variance
	d = np.sqrt(np.pi * denom)
	num = np.array(num)
	g = (np.exp(-num / denom) )/ d
	if ord == 1:
		g = -g * (x / variance)
	elif ord == 2:
		g = g * ((num / variance) / (variance ** 2))
	return g

def normalize(f):
	f = f - np.mean(f)
	f = f / sum(sum(np.absolute(f)))
	return f

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
def generate_half_disks():
	base = np.zeros([11,11])
	cv2.circle(base, (5,5), 5, (255,255,255), -1)
	base2 = base.copy()
	base[:,5:] = 0
	base2[:,:5] = 0
	base3 = np.zeros([21, 21])
	cv2.circle(base3, (10, 10), 10, (255, 255, 255), -1)
	base4 = base3.copy()
	base3[:, 10:] = 0
	base4[:, :10] = 0
	base5 = np.zeros([31, 31])
	cv2.circle(base5, (15, 15), 15, (255, 255, 255), -1)
	base6 = base5.copy()
	base5[:, 15:] = 0
	base6[:, :15] = 0
	orients = np.linspace(0, 180, 8, endpoint = False)
	masks=[]
	for i, orient in enumerate(orients):
		# imageFil=skimage.transform.rotate(conFilter, orient)
		# filters.append(imageFil)
		# imageFil=0
		rows1, cols1 = base5.shape
		M = cv2.getRotationMatrix2D((cols1 / 2, rows1 / 2), orient, 1)
		dst = cv2.warpAffine(base5, M, (cols1, rows1))
		dst2 = cv2.warpAffine(base6, M, (cols1, rows1))
		masks.append(dst)
		masks.append(dst2)
	for i, orient in enumerate(orients):
		# imageFil=skimage.transform.rotate(conFilter, orient)
		# filters.append(imageFil)
		# imageFil=0
		rows1, cols1 = base.shape
		M = cv2.getRotationMatrix2D((cols1 / 2, rows1 / 2), orient, 1)
		dst = cv2.warpAffine(base, M, (cols1, rows1))
		dst2 = cv2.warpAffine(base2, M, (cols1, rows1))
		masks.append(dst)
		masks.append(dst2)
	for i, orient in enumerate(orients):
		# imageFil=skimage.transform.rotate(conFilter, orient)
		# filters.append(imageFil)
		# imageFil=0
		rows2, cols2 = base3.shape
		M2 = cv2.getRotationMatrix2D((cols2 / 2, rows2 / 2), orient, 1)
		dst3 = cv2.warpAffine(base3, M2, (cols2, rows2))
		dst4 = cv2.warpAffine(base4, M2, (cols2, rows2))
		masks.append(dst3)
		masks.append(dst4)
	plot_DoG(masks,'masks.png')
	return masks

	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""


	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Brightness Map
	Perform brightness binning 
	"""


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Color Map
	Perform color binning or clustering
	"""


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""

#TODO: rework algo if possible
def generate_kernel(scales):
	kernels = []
	kernlen = 49
	for nsig in scales:
		interval = (2*nsig+1.)/(kernlen)
		x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
		kern1d = np.diff(st.norm.cdf(x))
		kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
		kernels.append(kernel_raw/kernel_raw.sum())
		# x, y = np.meshgrid(np.linspace(-s,s,50), np.linspace(-s,s,50))
		# d = np.sqrt(x*x+y*y)
		# sigma, mu = 1.0, 0.0
		# kernels.append(np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ))
		#kernels.append(cv2.getGaussianKernel(s, 1))
	return kernels

def gabor(sigmas, theta, Lambda, psi, gamma):
	gFilters = []
	for sigma in sigmas:
		"""Gabor feature extraction."""
		sigma_x = sigma
		sigma_y = float(sigma) / gamma

		# Bounding box
		nstds = 3  # Number of standard deviation sigma
		xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
		xmax = np.ceil(max(1, xmax))
		ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
		ymax = np.ceil(max(1, ymax))
		xmin = -xmax
		ymin = -ymax
		(y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

		# Rotation
		x_theta = x * np.cos(theta) + y * np.sin(theta)
		y_theta = -x * np.sin(theta) + y * np.cos(theta)

		gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
		gFilters.append(gb)
		gb = 0
	return gFilters

def combine_filters(img, filter1, filter2, filter3, filename):
	x, y = img.shape
	print(img.shape)
	texton_map = np.dstack((filter1,filter2,filter3))
	m, n, r = texton_map.shape
	print(texton_map.shape)
	res = np.reshape(texton_map, ((x * y), r))
	kmeans = sklearn.cluster.KMeans(n_clusters=64, random_state=2)
	kmeans.fit(res)
	labels = kmeans.predict(res)
	l = np.reshape(labels, (m, n))
	plt.imshow(l)
	plt.savefig(filename)
	plt.close()
	return l

def apply_filters(filters, img):
	texton_map = np.array(img)
	for filter in filters:
		result = cv2.filter2D(img, -1, filter)
		texton_map = np.dstack((texton_map, result))
	return texton_map

#TODO: use cv2.imwrite
def plot_DoG(filters, fileName, LM=False):
	if LM:
		_,_,r = filters.shape
	else:
		r = len(filters)
	plt.subplots(r/8,8,figsize=(5,5))
	for i in range(r):
		plt.subplot(r/8,8,i+1)
		plt.axis('off')
		if LM:
			plt.imshow(filters[:,:,i],cmap='gray')
		else:
			plt.imshow(filters[i],cmap='gray')
	plt.savefig(fileName)
	plt.close()

def plot_LM(filters, fileName):
    _,_,r = filters.shape
    plt.subplots(4,12,figsize=(20,20))
    for i in range(r):
        plt.subplot(4,12,i+1)
        plt.axis('off')
        plt.imshow(filters[:,:,i],cmap='binary')
    plt.savefig(fileName)
    plt.close()

def brightnessColor(img, x, y, z, filename):
	res = np.reshape(img, ((x * y), z))
	kmeans = sklearn.cluster.KMeans(n_clusters=16, random_state=2)
	kmeans.fit(res)
	labels = kmeans.predict(res)
	l = np.reshape(labels, (x, y))
	plt.imshow(l)
	plt.savefig(filename)
	plt.close()
	return l

def binary(img,bin_value):
    binary_img = img * 0
    for r in range(0,img.shape[0]):
        for c in range(0,img.shape[1]):
            if img[r, c]==bin_value:
                binary_img[r, c] = 1
            else:
                binary_img[r, c] = 0
    return binary_img

def chi_sqr(img, num_bins, left_mask, right_mask):
	chi_sqr_dist = img*0
	for i in range(num_bins):
		tmp = img.copy()
		tmp[img == i] = 1
		tmp[img != i] = 0
		tmp = tmp.astype('float64')
		g_i = cv2.filter2D(tmp, -1, left_mask)
		h_i = cv2.filter2D(tmp, -1, right_mask)
		chi_sqr_dist = chi_sqr_dist + np.divide(((g_i - h_i)**2),(g_i + h_i + np.exp(10**-7)))
	return chi_sqr_dist/2

def generate_gradient(filter, num_bins, masks):
	gradient = np.array(filter)
	for i in range(len(masks)/2):
		grad = chi_sqr(filter, num_bins, masks[2*i], masks[2*i + 1])
		gradient = np.dstack((gradient, grad))
	result = np.mean(gradient,axis = 2)
	return result


if __name__ == '__main__':
	main()
 


