import pickle
import numpy as np
import tensorflow
import cv2
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
feature_list=np.array(pickle.load(open('embeddings.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))



model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

img = image.load_img('sample/shirt.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img, axis=0)
preprocess_img = preprocess_input(expanded_img_array)
result = model.predict(preprocess_img).flatten()
nornalized_result = result / norm(result)

from sklearn.neighbors import  NearestNeighbors
neighbors=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)
distances,indices=neighbors.kneighbors([nornalized_result])

print(indices)

for file in indices[0][1:6]:
    temp_img=cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(170,170)))
    cv2.waitKey(0)

