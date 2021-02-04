from cv2.dnn import blobFromImage, readNetFromCaffe
from ImageDetectors.Models.Constants import FACE_DETECTOR_DIR, MIN_CONFIDENCE
from os.path import join
from cv2 import resize
import numpy as np

FACE_DETECTOR_DEFINITION = join(FACE_DETECTOR_DIR, 'deploy.prototxt')
FACE_DETECTOR_WEIGHTS = join(FACE_DETECTOR_DIR, 'weights.caffemodel')
INPUT_SIZE = (300,300)
TRAINING_MEAN = (104.0, 177.0, 123.0)

# Positions
CONFIDENCE_POS = 2
X2, Y2 = -2, -1


class OpenCVFaceDetector:
    """
    Convenient implementation of the Face Detection model offered by OpenCV.
    """
    def __init__(self):
        self.network = readNetFromCaffe(prototxt=FACE_DETECTOR_DEFINITION, caffeModel=FACE_DETECTOR_WEIGHTS)
        self.input_size = INPUT_SIZE

    def predict(self, rgb_uint8_img):
        """
        Take an rgb image in uint8 format as an input and returns the boxes where it is more likely to be a face.
        :param rgb_uint8_img: Numpy HxWx3 uint8 image. Image where to locate faces
        :return: List of boxes in format x1, y1, x2, y2.
        """
        h, w = rgb_uint8_img.shape[:2]
        blob = blobFromImage(resize(rgb_uint8_img, self.input_size), scalefactor=1.0,
                             size=self.input_size,
                             mean=TRAINING_MEAN, swapRB=True)
        self.network.setInput(blob)
        detections = self.network.forward()
        boxes = self.process_output(output=detections, h=h, w=w)
        return boxes


    def process_output(self, output, h, w):
        """
        Take the boxes into the model output format and clean of it all boxes that have no sense or very low confidence.
        Also transform them from Normalized format [0.,1.] into the image format [0, width|height]
        :param output: Numpy Array. output of the Face Detector model.
        :param h: Int. Height of the image.
        :param w: Int. Weight of the image.
        :return: List of Boxes in format x1, y1, x2, y2. Every boxes represents a detected face
        """
        # Take only those with confidence above the threshold
        output = output[output[..., CONFIDENCE_POS] > MIN_CONFIDENCE]
        # Take only those whose right side is lower than 1.0 (Values higher than 1.0 it tends to be detection errors)
        output = output[output[...,X2] <= 1.0]
        output = output[output[..., Y2] <= 1.0]
        # Clips the output into 0., 1. for ensuring that it size is correct
        output = np.clip(output,a_min=0., a_max=1.)
        # Return them normalized into the image space
        return [(face[CONFIDENCE_POS + 1:] * np.array([w, h, w, h])).astype(np.int) for face in output]

    def get_cropped_faces(self, rgb_uint8_image, margin=0.2, as_square_as_posible = True, boxes=None, return_box_position=False):
        """
        Return the subimages of the given rgb_uint8_image that should contain a face.
        :param rgb_uint8_image: rgb_uint8_img: Numpy HxWx3 uint8 rgb_uint8_image. Image where to locate faces.
        :param margin: Float. Margin to apply around the detected boxes for taking more face context. Default: 0.2
        :param as_square_as_posible: Boolean. If True, the cropping step will try to give an square output. It
        is useful for those cases where the face extraction is part of a Pipeline, where the following models
        do expect square inputs. It avoid future deformations. Default: True.
        :param boxes: List of boxes in format x1, y1, x2, y2. If not given, they will be predicted. Default None.
        :param return_box_position: Boolean. If true, also return the position of the boxes with all paddings included.
        :return: List of HxWx3 uint8 subimages containing the faces in the given rgb_uint8_image. If return_box_position
        flag is True, also returns the correspondant boxes.
        """
        if boxes is None:
            boxes = self.predict(rgb_uint8_img=rgb_uint8_image)
        # Calculate which will be the paddings to the box
        margins = [((x2-x1)*margin, (y2-y1)*margin) for (x1, y1, x2, y2) in boxes]
        # Transform these pads to format x1, y1, x2, y2
        padded_points = [(max(int(x1-x_marg), 0), max(int(y1-y_marg), 0),
                          min(int(x2+x_marg), rgb_uint8_image.shape[1]), min(int(y2 + y_marg), rgb_uint8_image.shape[0]))
                        for (x1, y1, x2, y2), (x_marg, y_marg) in zip(boxes, margins)]

        # Transform those values to square, if the borders of image do not interfere
        if as_square_as_posible:
            common_pad = [max(y2-y1, x2-x1) for x1, y1, x2, y2 in padded_points]
            padded_points = [(x1, y1, min(x1 + pad, rgb_uint8_image.shape[1]), min(y1 + pad, rgb_uint8_image.shape[0]))
                            for pad, (x1, y1, x2, y2) in zip(common_pad, padded_points)]
        # Crop the faces
        faces = [rgb_uint8_image[y1:y2, x1:x2] for x1, y1, x2, y2 in padded_points]

        # Return them with or without the corresponding boxes
        if return_box_position:
            return faces, padded_points
        else:
            return faces

