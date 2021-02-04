from cv2.dnn import blobFromImage, readNetFromCaffe
from ImageDetectors.Models.Constants import AGE_ESTIMATOR_DIR, MIN_CONFIDENCE, DECIMALS
from os.path import join
from cv2 import resize, cvtColor, COLOR_BGR2RGB
import numpy as np

AGE_ESTIMATOR_DEFINITION = join(AGE_ESTIMATOR_DIR, 'deploy.prototxt')
AGE_DETECTOR_WEIGHTS = join(AGE_ESTIMATOR_DIR, 'weights.caffemodel')
INPUT_SIZE = (224,224)
#TRAINING_MEAN = (104.0, 177.0, 123.0)

# Positions
CONFIDENCE = 2
X2, Y2 = -2, -1


class AgeEstimator:
    def __init__(self):
        self.network = readNetFromCaffe(prototxt=AGE_ESTIMATOR_DEFINITION, caffeModel=AGE_DETECTOR_WEIGHTS)
        self.input_size = INPUT_SIZE

    def predict(self, face):
            blob = blobFromImage(resize(face, self.input_size), scalefactor=1.0,
                                 size=self.input_size, swapRB=True)
            self.network.setInput(blob)
            probabilities = self.network.forward()[0]
            estimation = probabilities.dot(np.arange(len(probabilities)))
            return round(estimation, ndigits=DECIMALS)