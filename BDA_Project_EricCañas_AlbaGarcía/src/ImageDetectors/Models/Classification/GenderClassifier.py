from cv2.dnn import blobFromImage, readNetFromCaffe
from ImageDetectors.Models.Constants import GENDER_ESTIMATOR_DIR, MIN_CONFIDENCE
from os.path import join
from cv2 import resize

GENDER_ESTIMATOR_DEFINITION = join(GENDER_ESTIMATOR_DIR, 'deploy.prototxt')
GENDER_DETECTOR_WEIGHTS = join(GENDER_ESTIMATOR_DIR, 'weights.caffemodel')
INPUT_SIZE = (224,224)

# Positions
CONFIDENCE = 2
X2, Y2 = -2, -1
GENDER = ['Female', 'Male']

class GenderEstimator:
    def __init__(self):
        self.network = readNetFromCaffe(prototxt=GENDER_ESTIMATOR_DEFINITION, caffeModel=GENDER_DETECTOR_WEIGHTS)
        self.input_size = INPUT_SIZE

    def predict(self, face):
        blob = blobFromImage(resize(face, self.input_size), scalefactor=1.0,
                             size=self.input_size, swapRB=True)
        self.network.setInput(blob)
        probabilities = self.network.forward()[0]
        gender = GENDER[probabilities.argmax()]
        return gender