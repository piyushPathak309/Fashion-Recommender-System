import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
import numpy as np
import os
import pickle
from tqdm import tqdm
from numpy.linalg import norm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])


def Extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    nornalized_result = result / norm(result)

    return nornalized_result


filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

feature_list = []
for file in tqdm(filenames):
    feature_list.append(Extract_features(file, model))
print(np.array(feature_list).shape)

#
# pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
# pickle.dump(filenames, open('filenames.pkl', 'wb'))
