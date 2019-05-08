import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def getFeatureSpace(Img):

    featureSpace = np.ones((256, 256, 256))    # 2d array of r and g componenets
    for y in range(Img.shape[0]):
        for x in range(Img.shape[1]):
            featureSpace[Img[y,x,0] , Img[y,x,1], Img[y,x,2]] = 0
    return featureSpace


def Kmeans(originalImg,k,iterationNum):

    featureSpaceImg= getFeatureSpace(originalImg)
    means = []  # 2d array of mean values (y,x)
    for i in range(k):
        means.append([np.random.randint(0, featureSpaceImg.shape[0]), np.random.randint(0, featureSpaceImg.shape[1]), np.random.randint(0, featureSpaceImg.shape[2])])
    distances = np.zeros(k)



    for i in range(iterationNum):
        clusterColors = []
        clusterSum = []
        clusterCount = []
        #plt.imshow(featureSpaceImg, cmap='gray')
        for clusterNum in range(k):

            clusterSum.append([0,0,0])
            clusterColors.append([abs(np.random.rand()), abs(np.random.rand()), abs(np.random.rand())])
            clusterCount.append([0,0,0])
            #print(clusterColors[clusterNum])
            #plt.scatter(means[clusterNum][0], means[clusterNum][1], marker='x', c='b', s=100)

        #plt.show()
        #time.sleep(0.5)
        for z in range(featureSpaceImg.shape[0]):
            for y in range(featureSpaceImg.shape[1]):
                for x in range(featureSpaceImg.shape[2]):
                    if featureSpaceImg[z,y,x]==0 :

                        for clusterNum in range(k):
                            distances[clusterNum]=np.round(np.sqrt((z - means[clusterNum][0]) ** 2 +(y - means[clusterNum][1]) ** 2 + (x - means[clusterNum][2]) ** 2))
                        #print(means)
                        #print(means[clusterNum][0])
                        pixelClusterNum=np.argmin(distances, axis = 0)
                        clusterSum[pixelClusterNum]=np.add(clusterSum[pixelClusterNum],[z,y,x])
                        #print(clusterSum)
                        clusterCount[pixelClusterNum]=np.add(clusterCount[pixelClusterNum],[1,1,1])
                        #print(clusterCount[pixelClusterNum])
                        #print(clusterSum[pixelClusterNum])
                        #plt.scatter(y, x, marker='.', c=clusterColors[pixelClusterNum], s=1)
        '''
        print("*********************************")
        print(means)
        print(clusterSum)
        print(clusterCount)
        print("*********************************")
        '''
        for clusterNum in range(k):
            if(clusterCount[clusterNum][0]> 0):
                means[clusterNum]=np.divide(clusterSum[clusterNum],clusterCount[clusterNum])
                means[clusterNum][0] = int(means[clusterNum][0])
                means[clusterNum][1] = int(means[clusterNum][1])
                means[clusterNum][2] = int(means[clusterNum][2])

        #means=int(np.divide(clusterSum,clusterCount))
        #plt.show()
    #print(means)
    img=np.zeros((originalImg.shape[0],originalImg.shape[1],3), 'int8')
    medianArray=np.zeros((64,64,3), 'int16')
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):

            for clusterNum in range(k):
                distances[clusterNum] = np.round(np.sqrt((originalImg[y,x,0] - means[clusterNum][0]) ** 2 + (originalImg[y,x,1] - means[clusterNum][1]) ** 2+ (originalImg[y,x,2] - means[clusterNum][2]) ** 2))
            pixelClusterNum = np.argmin(distances, axis=0)

            #img[y,x]= clusterColors[pixelClusterNum]
            #img[y,x] = np.multiply(clusterColors[pixelClusterNum],255)
            img[y, x] = means[pixelClusterNum]
            #print(img[y, x],means[pixelClusterNum])
            if 31<y<96 and 31<x<96:
                medianArray[y-32, x-32] = means[pixelClusterNum]
                #print(medianArray[y, x], means[pixelClusterNum])
                #return means[pixelClusterNum][0],means[pixelClusterNum][1],means[pixelClusterNum][2]
        #    plt.scatter(y, x, marker='.', c=clusterColors[pixelClusterNum], s=1) round(means[pixelClusterNum][0],2)

    '''plt.imshow(featureSpaceImg, cmap='gray')
    for clusterNum in range(k):
        plt.scatter(means[clusterNum][1], means[clusterNum][0], marker='x', c='r', s=100)
    plt.show()
    '''
    #print(np.multiply(clusterColors, 255))
    #img = img[32:96, 32:96]
    median=np.median(medianArray[:,:,0])
    #print(median)

    img = Image.fromarray(img, 'RGB')
    plt.imshow(img)
    plt.show()
    for clusterNum in range(k):
        if int(means[clusterNum][0])==int(median):
            return means[clusterNum][0],means[clusterNum][1],means[clusterNum][2]

'''
features=np.zeros([210, 4], dtype = float)

a=np.genfromtxt("./flower_images/flower_labels.csv", delimiter=',')
a=a[1::,1]
j=0
for i in range(210):
     PATH = "./flower_images/0001.jpg"
     N = "%03d" % (i+1)
     if a[i]==0 or a[i]==4 or a[i]==2 or a[i]==5 or a[i]==7 :

         print (N)
         img = mpimg.imread(PATH.replace("001", N))
         features[j][0],features[j][1],features[j][2]=Kmeans(img,2,5)
         features[j][3]=a[i]
         j=j+1


np.savetxt("features.csv", features,fmt='%.2f', delimiter=",")

print( np.genfromtxt('features.csv', delimiter=','))
'''


#img = mpimg.imread("./flower_images/0012.jpg")
#featureSpace = getFeatureSpace(img)
#plt.imshow(img[32:96,32:96])
#plt.show()
#print(featureSpace)

#print(Kmeans(img,2,5))

#plt.imshow(getFeatureSpace(img), cmap='gray')
#plt.show()
