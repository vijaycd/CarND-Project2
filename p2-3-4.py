#Vijay D, Udacity CarND- Term1, Project 2: Traffic Sign Classifer

# Load training, validation and test data (pickled data)
import pickle
from itertools import *
import numpy as np

training_file = 'data/train.p'
validation_file='data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
unique_classes =  len(np.unique(y_train))

#print('Total training examples = ', len(X_train) )
#print('Total testing examples  = ', len(X_test) )
#print('# of images with shape  = ', X_train.shape)
#print("# of classes =", unique_classes)
#print ('y_train', type(y_train))
#print ('y_train shape', y_train.shape)
#print ('X_train', type(X_train))

#x = islice(train.items(), 0, 4)
#for key, value in x:
#    print (key, value)

#Visualize Data
#View a sample from the dataset.
import random
import matplotlib.pyplot as plt
#%matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

#plt.figure(figsize=(1,1))
#plt.imshow(image)
#print(y_train[index])
#print(image.shape)

train_images, train_labels = X_train, y_train
h = np.histogram(train_labels, bins=np.arange(unique_classes))
#print (h[0], h[1])

# the histogram of the data
#n, bins, patches = plt.hist(h[1])

#plt.xlabel('Unique Classes')
#plt.ylabel('# of images')
#plt.title('Image distribution in Training Data')
#end = 42
##plt.plot(h[1][:end], h[0])
#plt.figure(figsize=(30,20))
#plt.hist(h[0], bins=43, histtype='step')
#plt.show()

#import plotly.plotly as py
#from plotly.graph_objs import *

#data = [Bar(x=h[1],
#            y=h[0])]

#py.iplot(data, filename='basic_bar')

#Preprocessing work done further below. Only normalization and shuffling done here.
def preprocess(image_batch):
    
    for eachimg in image_batch:
        #print (eachimg.shape)
        eachimg = cv2.cvtColor(eachimg, cv2.COLOR_RGB2GRAY)
        #print (eachimg.shape)
        eachimg = (eachimg-128.)/128.
   
    return image_batch

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

############################DATA AUGMENTATION########################################
#Some data augmentation done here...thru brightness adj, rotation, translation.
import cv2
augmented_images = 0

# a la Alex @ navoshta.com
def flip_extend(X, y):
    # Classes of signs that, when flipped horizontally, should still be classified as the same class
    self_flippable_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    # Classes of signs that, when flipped vertically, should still be classified as the same class
    self_flippable_vertically = np.array([1, 5, 12, 15, 17])
    # Classes of signs that, when flipped horizontally and then vertically, should still be classified as the same class
    self_flippable_both = np.array([32, 40])
    # Classes of signs that, when flipped horizontally, would still be meaningful, but should be classified as some other class
    cross_flippable = np.array([
        [19, 20], 
        [33, 34], 
        [36, 37], 
        [38, 39],
        [20, 19], 
        [34, 33], 
        [37, 36], 
        [39, 38],   
    ])
    num_classes = 43
    
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = X.dtype)
    y_extended = np.empty([0], dtype = y.dtype)
    
    for c in range(num_classes):
        # First copy existing data for this class
        X_extended = np.append(X_extended, X[y == c], axis = 0)
        # If we can flip images of this class horizontally and they would still belong to said class...
        if c in self_flippable_horizontally:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X[y == c][:, :, ::-1, :], axis = 0)
        # If we can flip images of this class horizontally and they would belong to other class...
        if c in cross_flippable[:, 0]:
            # ...Copy flipped images of that other class to the extended array.
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(X_extended, X[y == flip_class][:, :, ::-1, :], axis = 0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
        
        # If we can flip images of this class vertically and they would still belong to said class...
        if c in self_flippable_vertically:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, :, :], axis = 0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
        
        # If we can flip images of this class horizontally AND vertically and they would still belong to said class...
        if c in self_flippable_both:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis = 0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
    
    return (X_extended, y_extended)



# a la Vivek Yadav and OpenCV.org
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
	
	
# a la Vivek Yadav and OpenCV.org
def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img

# a la Vivek Yadav and OpenCV.org
def DisplayGrid(X):	
	gs1 = gridspec.GridSpec(25, 20)
	gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
	plt.figure(figsize=(17,17))

	freq = 1
	#Display grid
	for i in range(len(X)):
			#print (y_train[i], clnum, (y_train[i]==clnum))		
			ax1 = plt.subplot(gs1[i])
			ax1.set_xticklabels([])
			ax1.set_yticklabels([])
			ax1.set_aspect('equal')
			img = X[i].squeeze()
			plt.subplot(25,20,i+1)
			plt.imshow(img)
			plt.axis('off')
			#print ('Found image of class', clnum, 'Freq', freq)
			freq+=1
	plt.show()

def GetIndexof(lbl):
	for i in range(len(unique_labels)):

		if unique_labels[i] == lbl:
			return i


unique_labels = np.unique(y_train)

#Plotting number of samples against classes
counts = []
for i in range(len(unique_labels)):
    counts.append(np.sum(y_train == i))

print('Total training set examples \t= ', len(X_train) )
print('Shape of training set  \t\t= ', X_train.shape)
print ('Total # of classes \t\t= ', len(unique_labels))
	
#Find mean and std dev of the existing distribution
mu = np.mean(counts)
sigma = np.std(counts, dtype=np.float64)
print ('Image Class Distribution \n\tMean \t\t\t=  {:.2f}'.format(mu), '\n\tStandard Deviation \t=  {:.2f}'.format(sigma ))

#plt.bar(range(len(counts)), counts, align='center', alpha=0.4)
#plt.title('Original Training Dataset Class Distribution')
#plt.ylabel('# of images')
#plt.show()

# Temp placeholder lists for augmented data
X = [] 
y = []

for label in unique_labels:
		
		label_index = GetIndexof(label)
		#print ('value of counts at index:', label_index, counts[label_index])
		for i in range(len(X_train[0])):
			if (y_train[i] == label):
				image = X_train[i].squeeze()
				#plt.imshow(image);
				#plt.axis('off');

		#gs1 = gridspec.GridSpec(10, 10)
		#gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
		#plt.figure(figsize=(13,13))

		#To equalize the distribution of classes approx., we will bring up each class to a total of (mu+sigma)
		images_needed = int(mu + 2*sigma - counts[label_index])
		#print ('Images needed for class', label, images_needed)
		if (images_needed > 0): #Augment only when a class is underrepresented
			augmented_images += images_needed
			for i in range(images_needed):
				img = transform_image(image,20,10,5,brightness=1)
				X.append(img)
				y.append(label)

#Converting lists to arrays
X_tend = np.array(X, dtype=np.float32)
y_tend = np.array(y, dtype=np.int32)

#DisplayGrid(X_tend)  -- not used 

#Append the augmented data arrays to X_train, y_train, etc.
X_train = np.concatenate((X_train, X_tend), axis=0)
y_train = np.concatenate((y_train, y_tend), axis=0)

counts1 = []
for i in range(len(unique_labels)):
    counts1.append(np.sum(y_train == i))
#print (counts1, counts1[0] )

#plt.bar(range(len(counts1)), counts1, align='center', alpha=0.4)
#plt.title('Augmented Training Dataset Class Distribution')
#plt.ylabel('# of images')
#plt.show()

#Find mean and std dev of the augmented dataset
mu = np.mean(counts1)
sigma = np.std(counts1, dtype=np.float64)
#print ('Augmented Data Mean, Std. Dev:', mu, sigma)

#print ('Total length of training dataset:', len(X_train))
print ('Images augmented via rotation/translation/brighness: ', augmented_images, '\nfor a total of: ', len(y_train), 'images\n')

#Shuffle unamas!
X_train, y_train = shuffle(X_train, y_train)

#############################IMAGE FLIPPING##########################################
X_train, y_train = flip_extend(X_train, y_train)
print ("Images augmented by flipping: ", len(X_train))

############################END AUGMENTATION#########################################

import tensorflow as tf

EPOCHS = int(input("Epochs? "))
#Fixed below to 64 based on findings in https://arxiv.org/abs/1609.04836 ("On Large-Batch Training for Deep Learning")
BATCH_SIZE = 50 #int(input ("Batch size?"))


from tensorflow.contrib.layers import flatten

def SimpLeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    dropout = 0.5
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x32.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 32), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: New Layer 1: Convolutional. Input = 10x10x32. O/P = 6x6x64
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(64))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    # SOLUTION: Activation for NL 1
    conv3 = tf.nn.relu(conv3)
        
    # SOLUTION: Pooling. Input = 6x6x64. Output = 3x3x64.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 3x3x64. Output = 576
    fc0   = flatten(conv3)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 576. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(576, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
        
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    #Adding a new dropout layer 1
    fc1    = tf.nn.dropout(fc1, dropout)
    
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    #Adding a new dropout layer 2
    fc2    = tf.nn.dropout(fc2, dropout)
    
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
	
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
	
rate = 0.001

logits = SimpLeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
	
from time import time, localtime, strftime

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    keep_track = [] #For tracking accuracies by epoch
	
    print("Training... Batchsize = ",BATCH_SIZE, '| # of Epochs = ', EPOCHS)
    print()
    t1 = time()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            preprocess(batch_x) #new
            #print(batch_x.shape, batch_y.shape)
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        t1t = time()
        print ('Elapsed time:  {:.2f}'.format((t1t-t1)/60.), 'min.')
        print()
        keep_track.append(validation_accuracy)
        
    saver.save(sess, './lenet')
    print("Model saved")
    t2 = time()
    print ('Total processing time \t: {:.2f}'.format((t2-t1)/60.), 'min.')
    vmax = np.max(keep_track)
    print ('Maximum Accuracy \t: {:.2f}'.format(vmax*100.), 'at ', keep_track.index(vmax))
	