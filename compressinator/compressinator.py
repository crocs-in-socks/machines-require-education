import matplotlib.pyplot as plt
import numpy as np
# import cv2
from skimage import io

class kMeansClustering:

    def __init__(self, points, noOfClusters, noOfIterations):

        self.noOfClusters = noOfClusters
        self.noOfPoints, self.noOfColors = points.shape
        self.points = points
        self.noOfIterations = noOfIterations


    def squaredDistance(self, x1, y1, x2, y2):
        return np.square(x1 - x2) + np.square(y1 - y2)
    

    def initCentroids(self):

        print("Intializing centroids.")
        centroids = np.zeros((self.noOfClusters, self.noOfColors))

        for cluster in range(self.noOfClusters):
            centroids[cluster] = self.points[np.random.choice(self.noOfPoints)]
        
        print("Centroids Initialized.")

        return centroids
    

    def calculateCost(self, centroids, clusterIdx):

        cost = 0
        
        for cluster in range(self.noOfClusters):

            pointsInCluster = self.points[clusterIdx == cluster]

            for point in range(len(pointsInCluster)):

                x1, y1 = pointsInCluster[point, 0], pointsInCluster[point, 1]
                x2, y2 = centroids[cluster, 0], centroids[cluster, 1]
                cost += self.squaredDistance(x1, y1, x2, y2)
            
        print(f"Current cost : {cost}")

        return cost
        

    def runKMeans(self):

        centroids = self.initCentroids()
        clusterIdx = np.zeros(self.noOfPoints)
        J = []

        while True:

            oldCentroids = centroids

            # Finding out which centroid is the closest for all points
            for point in range(self.noOfPoints):

                minDistance = float('inf')

                # Checking distance from each of the centroids
                for cluster in range(self.noOfClusters):

                    x1, y1 = self.points[point, 0], self.points[point, 1]
                    x2, y2 = centroids[cluster, 0], centroids[cluster, 1]

                    # Update if current centroid is closer
                    if self.squaredDistance(x1, y1, x2, y2) <= minDistance:
                        minDistance = self.squaredDistance(x1, y1, x2, y2)
                        clusterIdx[point] = cluster
            
            print("Iterated through all points.")
        
            for cluster in range(self.noOfClusters):

                pointsInCluster = self.points[clusterIdx == cluster]
                
                if len(pointsInCluster) > 0:
                    # axis=0 means column-wise
                    centroids[cluster] = np.mean(pointsInCluster, axis=0)
            
            print("Centroids adjusted.")
            # print(f"Centroids : {centroids}")
        
            J.append(self.calculateCost(centroids, clusterIdx))

            if oldCentroids.all() == centroids.all():
                break

        return centroids, clusterIdx, J
    
    
    def returnClusters(self):

        minJ = [float('inf')]

        for iteration in range(self.noOfIterations):
            
            print()
            print(f"Starting iteration #{iteration+1}")
            centroids, clusterIdx, J = self.runKMeans()

            if J[-1] < minJ[-1]:
                bestCentroids = centroids
                bestClusterIdx = clusterIdx
                minJ = J
            
            print(f"Iteration #{iteration+1} done, Cost : {J[-1]}")
            print(f"Minimum cost so far : {minJ[-1]}")
            print()

        return bestCentroids, bestClusterIdx


def readImage(imagePath):
    image = io.imread(imagePath)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalizing the image
    image = image/255

    return image

def compressImage(imagePath):
    image = readImage(imagePath)
    colors = 16
    iterations = 10

    points = image.reshape((-1, image.shape[2]))

    print("Compressing the image.")

    kMeans = kMeansClustering(points, colors, iterations)
    centroids, clusterIdx = kMeans.returnClusters()

    # Here, every point in clusterIdx is used as an index to centroids, from which the compressed array is generated
    compressedImage = centroids[clusterIdx.astype(int), :]
    compressedImage = compressedImage * 255
    compressedImage = compressedImage.astype('uint8')
    # compressedImage = np.clip(compressedImage.astype('uint8'), 0, 255)

    # with np.printoptions(threshold=np.inf):
    #     print(f"Compressed Image : {compressedImage[0]}")

    compressedImage = compressedImage.reshape(image.shape)
    # plt.imshow(compressedImage)
    # plt.show()

    print("Image compressed successfully.")
    io.imsave(f"peacock-{colors}.png", compressedImage)


if __name__ == "__main__":
    imagePath = "./peacock.png"
    compressImage(imagePath)