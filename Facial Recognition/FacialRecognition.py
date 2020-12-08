#!/usr/bin/env python
# coding: utf-8

# # <font color = 'blue'> Facial Recognition with Machine Learning</font>
# # <font color = 'blue'> Data Science Academy </font>
# 
# # <font color = 'blue'> Vinicius Biazon </font>

# ### Facial Recognition with Machine Learning Using SVM and PCA

# We are going to create a model for facial recognition, using SVM and PCA.
# 
# The dataset used in this project is the Labeled Faces in the Wild Home, a set of face images prepared for Computer Vision tasks. It is available both on Keras and at http://vis-www.cs.umass.edu/lfw/.

# ### Loading Packages

# In[1]:


# Image storage
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import svm

# Image Dataset
from sklearn import datasets

# Graph creation
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Loading dataset (at least 70 images per person with a 0.4 scaling factor)
dataset_faces = datasets.fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)


# In[3]:


# Checking the dataset shape
dataset_faces.data.shape


# ### Preparing the Dataset
# 

# In[4]:


# Extracting the shape details from the images
num_samples, height, width = dataset_faces.images.shape


# In[5]:


# Putting data in X (input variables) and target in y (output variable)
X = dataset_faces.data


# In[6]:


# Number of X attributes
num_attributes = X.shape[1]


# In[7]:


print(X)


# Each pixel can have a value from 0 to 255, for black and white images.

# In[8]:


# Putting the target in y
y = dataset_faces.target


# In[9]:


# Extracting class names
target_names = dataset_faces.target_names


# In[10]:


# Number of classes
num_class = target_names.shape[0]


# In[11]:


# Printing a summary of the data
print ("\nTotal Dataset Size: \n")
print ("Number of samples (images):% d"% num_samples)
print ("Height (pixels):% d"% height)
print ("Width (pixels):% d"% width)
print ("Number of Attributes (variables):% d"% num_attributes)
print ("Number of Classes (people):% d"% num_class)


# ### Viewing the Data

# In[12]:


# Images Plot

# Setting the plot area size
fig = plt.figure(figsize = (12, 8))

# 15 images Plot
for i in range(15):
    
    # Dividing images into 5 columns and 3 rows
    ax = fig.add_subplot(3, 5, i + 1, xticks = [], yticks = [])
    
    # Showing the images
    ax.imshow(dataset_faces.images[i], cmap = plt.cm.bone)


# ### Viewing the distribution of people from the Dataset

# In[13]:


# Setting the plot area size
plt.figure(figsize = (10, 2))

# Capturing unique target (class) values
unique_targets = np.unique(dataset_faces.target)

# Counting total of each class
counts = [(dataset_faces.target == i).sum() for i in unique_targets]

# Result plot
plt.xticks(unique_targets, dataset_faces.target_names[unique_targets])
locs, labels = plt.xticks()
plt.setp(labels, rotation = 45, size = 14)
_ = plt.bar(unique_targets, counts)


# ### Splitting the data in training and testing

# In[14]:


# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(dataset_faces.data, dataset_faces.target, random_state = 0)


# In[15]:


# Print
print(X_train.shape, X_test.shape)


# - For training we have 966 images and 1850 attributes, or images pixels.
# 
# 
# - For testing we have 322 images and 1850 attributes, or images pixels.

# ## Pre-Processing: Principal Component Analysis (PCA)

# We are going to use the PCA to reduce these 1850 resources to a manageable level, while keeping most of the information in the data set. We will create a PCA model with 150 components

# In[16]:


# Creating the PCA model
pca = decomposition.PCA(n_components = 150, 
                        whiten = True,
                        random_state = 1999, 
                        svd_solver = 'randomized')


# In[17]:


# Training the model
pca.fit(X_train)


# In[18]:


# Applying the PCA model to train and test data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# In[19]:


# Shape
print(X_train_pca.shape)
print(X_test_pca.shape)


# ### Creating the Machine Learning Model with SVM

# In[20]:


# Creating the model
svm_model = svm.SVC(C = 10., gamma = 0.001)


# In[21]:


# Training the model
svm_model.fit(X_train_pca, y_train)


# ### Evaluating the Model

# In[22]:


# Shape of test data
print(X_test.shape)


# In[23]:


# Plot area size
fig = plt.figure(figsize = (12, 8))

# 15 imagens loop
for i in range(15):
    
    # Subplots
    ax = fig.add_subplot(3, 5, i + 1, xticks = [], yticks = [])
    
    # Showing the real image in the test dataset
    ax.imshow(X_test[i].reshape((50, 37)), cmap = plt.cm.bone)
    
    # Making the class prediction with the trained model
    y_pred = svm_model.predict(X_test_pca[i].reshape(1,-1))[0]
    
    # Putting colors in the results
    color = 'black' if y_pred == y_test[i] else 'red'
    
    # Defining the title
    ax.set_title(dataset_faces.target_names[y_pred], fontsize = 'small', color = color)


# Red names represent model errors. Black names mean the model is right.

# ### Model Score

# In[24]:


print(svm_model.score(X_test_pca, y_test))


# This model has an efficiency around 84%, which means that for every 100 images the prediction is correct in 84 cases.
