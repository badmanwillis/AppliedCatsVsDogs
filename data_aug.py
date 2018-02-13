'''

Augmenting catsvsdogs training data using openCV, in place of imagedatagenerator


imagedatagenerator params
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())


focusing  on replacating

rotation, height and width shift, shear, zoom, horizontal and vertical flip, rescale
'''


#import keras
import cv2
import glob
import os
import random
import numpy as np

inputPath = sorted(list((glob.glob("/home/olly/projects/machine_learning/catsvsdogs/data/train/cats/*.jpg"))))
#path = "/media/olly/Storage/Uni/project/catsvsdogs_aug_data"
path = "/home/olly/Pictures/augmented_cats_test"


iteration = 0
for inputImages in inputPath:

	image = cv2.imread(inputImages)

	image = cv2.resize(image, (150, 150))

	
	# Roll for rotation, shear, flip, or blur
	# a better method that combines transforms may be needed
	choice = random.randint(1,4) # rot, trans, flip, blur



	# Press Esc to quit program
	
	k = cv2.waitKey(33)
	if k==27:    # Esc key to stop
		cv2.destroyAllWindows()
        	break
	


	# rotation
	if choice == 1:
		rows,cols = 150,150

		degree = random.randint(1,90)
		print ("rotation", degree)

		M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
		dst = cv2.warpAffine(image,M,(cols,rows))
	

		# Affine transformation
		# Applies subtle shear transforms
	if choice == 2:
		pts2Ax = random.randint((65-10),(65+10))
		pts2Ay = random.randint((85-10),(85+10))
		pts2Bx = random.randint((65-10),(65+10))
		pts2By = random.randint((45-10),(45+10))
		pts2Cx = random.randint((85-10),(85+10))
		pts2Cy = random.randint((85-10),(85+10))

		rows,cols,ch = image.shape
		pts1 = np.float32([[65,85],[65,45],[85,85]])
		pts2 = np.float32([[pts2Ax,pts2Ay],[pts2Bx,pts2By],[pts2Cx,pts2Cy]])


		#pts1 = np.float32([[pts1Ax,pts1Ay],[pts1Bx,pts1By],[pts1Cx,pts1Cy]])
		#pts2 = np.float32([[pts2Ax,pts2Ay],[pts2Bx,pts2By],[pts2Cx,pts2Cy]])

		#pts1 = np.float32([[50,50],[200,50],[50,200]])
		#pts2 = np.float32([[10,100],[200,50],[100,250]])
		M = cv2.getAffineTransform(pts1,pts2)
		dst = cv2.warpAffine(image,M,(cols,rows))
		#print pts1
		print ("transform",pts2)
		#print "\n"
	

	# height shift
	# width shift
	# zoom

	# horizontal flip
	if choice == 3:
	#flip = random.randint(0,1)
		print "flip"

	#if flip == 1:
		rows,cols,ch = image.shape
		pts1 = np.float32([[75,85],[75,45],[85,85]])
		pts2 = np.float32([[75,85],[75,45],[65,85]])


		#pts1 = np.float32([[pts1Ax,pts1Ay],[pts1Bx,pts1By],[pts1Cx,pts1Cy]])
		#pts2 = np.float32([[pts2Ax,pts2Ay],[pts2Bx,pts2By],[pts2Cx,pts2Cy]])

		#pts1 = np.float32([[50,50],[200,50],[50,200]])
		#pts2 = np.float32([[10,100],[200,50],[100,250]])
		M = cv2.getAffineTransform(pts1,pts2)
		dst = cv2.warpAffine(image,M,(cols,rows))


	# channel shift
	
	# Guassian blur
	if choice == 4:
		g = random.randint(0,7)*2+1 # random odd 3 - 15
		print ("blur", g)
		dst = cv2.GaussianBlur(image, (g,g),0)


	# Show original
	cv2.imshow("Cat", image)
	#cv2.waitKey(0)
	# Show modified image
	cv2.imshow("Aug", dst)
	cv2.waitKey(0)
	

	# Save augmented image
	iteration = iteration + 1
	print ("aug_data_%d.jpg" % iteration)
	cv2.imwrite(os.path.join(path,("aug_data_%d.jpg" % iteration)), dst)

	# Need to figure out how to iterate save image name, thereby saving lots of images, not overwriting one image.





























