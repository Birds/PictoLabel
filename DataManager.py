from imutils import paths
import random
import cv2
import os

class DataManager:
    def loadData(self, imagePaths):
        data = []
        labels = []
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (100, 100))
            data.append(image)
            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)

        return [data, labels]

    def loadDataPath(self, folder_name):
        listPaths = sorted(list(paths.list_images(folder_name)))
        random.seed(17)
        random.shuffle(listPaths)
        return listPaths