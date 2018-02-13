'''
2.2.18

The culmination of all learned thus far

Preparing the dataset: see dataset_prep

Model architecture: ripped from catsvsdogs

Using simple_MNIST as a refernce for a fully working model, with predictions.


If this works i'll be very happy.
'''

# DEPENDANCIES

# Data preperation
import cv2
import glob
from keras.preprocessing.image import img_to_array

# Data preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# DATASET PREPERATION

# initialize the data and labels
print("[INFO] loading images...\n")
data = []
labels = []

CatImagePaths = sorted(list((glob.glob("/home/olly/projects/machine_learning/catsvsdogs/data/train/cats/*.jpg"))))
DogImagePaths = sorted(list((glob.glob("/home/olly/projects/machine_learning/catsvsdogs/data/train/dogs/*.jpg"))))


for CatImages in CatImagePaths:

	image = cv2.imread(CatImages)

	image = cv2.resize(image, (150, 150))

	# Let's look at some lovely cats
	#cv2.imshow("Cat", image)
	#cv2.waitKey(0)

	# Store image in data list
	image = img_to_array(image)
	data.append(image)

	# Extract class label and update the labels list
	labels.append(0) 

for DogImages in DogImagePaths:

	image = cv2.imread(DogImages)

	image = cv2.resize(image, (150, 150))

	# Let's look at some lovely dogs
	#cv2.imshow("Dog", image)
	#cv2.waitKey(0)

	# Store image in data list
	image = img_to_array(image)
	data.append(image)

	# Extract class label and update the labels list
	labels.append(1) 

#print labels
# DATA PRE-PROCESSING
print("[INFO] data preprocessing...\n")


#		Dataset_prep Method

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(X_train, X_test, Y_train, Y_test) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
# Convert labels from ints to vectors, using One-hot encoding
# One-hot encoding perfoms "binarization" of the category, and includes it as a feature to train the model.
Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)



#		MNIST Method
'''
(X_train, X_test, Y_train, Y_test) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
from keras.utils import np_utils
# pre-processing

#X_train = X_train.reshape(X_train.shape[0], 3, 150, 150) # depth is 1, for B&W
#X_test = X_test.reshape(X_test.shape[0], 3, 150, 150)


X_train = X_train.reshape(None, 3, 150,150) # depth is 1, for B&W
X_test = X_test.reshape(None, 3, 150,150)
		

#print X_train.shape
# convert to float32 and normalise data values to the range [0,1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# Convert 1-dimensional class arrays to 2-dimensional class matrices
Y_train = np_utils.to_categorical(Y_train, 2)
Y_test = np_utils.to_categorical(Y_test, 2)
'''

# Data shapes
#print labels
#print ("Y_train",Y_train)
#print ("Y_test", Y_test)
print "Data shapes"
print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape



# The above from dataset_prep, simple_MNIST is slightly different, but code appears to work the same.







# MODEL ARCHITECTURE
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


img_width, img_height = 150, 150
input_shape = (3, img_width, img_height)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2)) # needs to be the size of the no. classes (eg 2, cats or dogs)
model.add(Activation('sigmoid'))

# TRAIN THE MODEL
print("[INFO] training model...\n")
# Compile
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# catsvsdogs uses binary_crossentropy, rmsprop, accuracy

# Fit model to training data
model.fit(X_train, Y_train, 
          batch_size=32, epochs=50, verbose=1)

# Evaluate
print "model score verbose 0"
score = model.evaluate(X_test, Y_test, verbose=0)
print score
print "model score verbos 1"
score = model.evaluate(X_test, Y_test, verbose=1)
print score
print("[INFO] model trained.\n")

# SAVE
print("[INFO] saving model...\n")
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print("[INFO] model saved.\n")

# MAKE A PREDICTION
# see predict_catsvsdogs2.py
'''	
print("[INFO] attempting prediction.\n")
# Attempt prediction
x_sample = X_test[0]#.reshape(None, 1, 28,28)
y_prob = model.predict(x_sample)[0]
y_pred = y_prob.argmax()
y_actual = y_test[0].argmax()

#plt.imshow(X_test[0])
#plt.show()

print("probabilities: ", y_prob)
print("predicted = %d, actual = %d" % (y_pred, y_actual))
'''
















