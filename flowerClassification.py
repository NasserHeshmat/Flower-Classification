import numpy as np
import matplotlib.image as mpimg
import Kmeans
from sklearn.externals import joblib


classifier = joblib.load('knnModel.pkl')
print(classifier)

img = mpimg.imread("./flower_images/0002.jpg")
newFeatures=np.zeros((1,3))
newFeatures[0][0],newFeatures[0][1],newFeatures[0][2] = Kmeans.Kmeans(img,2,5)
print("Image Features : ",newFeatures)

y_pred = classifier.predict(newFeatures)
print("Predicted Flower Class : ",y_pred[0])
