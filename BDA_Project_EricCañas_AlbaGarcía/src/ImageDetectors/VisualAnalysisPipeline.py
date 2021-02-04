from ImageDetectors.Models.Commons import *
from ImageDetectors.Models.Segmentation.Segmentator import Segmentator
from ImageDetectors.Models.Detection.FaceDetector import OpenCVFaceDetector
from Crawler.HashtagsCrawler.Constants import FACE_ESTIMATOR_MODELS
from ImageDetectors.Models.Commons import fuse_images_without_background
from ImageDetectors.Models.Detection.YOLOv5 import YOLOv5
import Database.Constants as FIELDS
from skimage.transform import resize
from PIL import Image

class Pipeline:
    """
    Feature extraction pipeline which implement all the Feature Extraction modules defined into Models
    in order to extract all the details within an image and to return them into a standard dictionary form.
    Note that, as this project is intended to be centered in Data Gathering and Data Science purposes, most
    Feature Extraction models are not trained from scratch, but borrowed by different authors and hubs that offered
    pretrained versions of them as open source tools. Every one of them is cited into the correspondent Citation.md.
    Please take a look at their open repositories if you also want to download them (and star them).
    Note that some of their implementations also does not come from scratch, since science is an extremely large
    pipeline where each one learn from the others and where each one focuses into a concrete part of the process
    for trying both to improve the existent open tools or to improve the general knowledge about any area where
    those tools could be useful.
    """
    def __init__(self):
        """
        Initiate all the Feature extraction models. Including ADE20K segmentator (Of keras-segmentation library),
        a Face Detector (offered by OpenCV), different estimators of face characteristics (Gender, Age and Expression
        (Oferred by their correspondent authors)), a Face Parsing Segmentator model (adapted to Keras and pretrained
        by its correspondent author), and the YOLOv5 object detector (offered its author through torch hub).
        """
        self.general_segmentator = Segmentator(ADE20K)
        self.face_detection = OpenCVFaceDetector()
        self.face_estimation = {name : model() for name, model in FACE_ESTIMATOR_MODELS.items()}
        self.face_segmentator = Segmentator(FACE_PARSING)
        self.object_counter = YOLOv5()

    def get_details_of_image(self, rgb_uint8_img, save_anonymized_img_as = None):
        """
        Process the image with all the feature extractors within the Pipeline for squeezing all the information within
        it.
        :param rgb_uint8_img: HxWx3 numpy image in uint8 format. Image from which squeeze all information.
        :param save_anonymized_img_as: String Path. Path where to save the anonymized segmentation map. If no path
         is given, the segmentation map is not saved. Default: None.
        :return: Dictionary with all details of the image. Being this details:
            For each object detected by the general segmentator: The object, its area, its median basic color and
            its median xkcd color. Ordered by area.
            For each face detected: The apparent age, the apparent gender and the apparent expression. And for each
            part of its head (eye, nose, mouth, ear, hat...) the same information than in the general segmentator.
            Also ordered by the area
            For each object detected by YOLOv5: The name of the object, the amount of objects of this type,
            the approximated area that these objects occupy. Also ordered by area.
            If save_anonymized_img_as path was given. A simplified segmentation path will have been saved.
        """
        # Initialize the dictionary
        details = {}
        # Include the general segmentation details
        general_segmentation_mask = self.general_segmentator.predict(rgb_uint8_image=rgb_uint8_img)
        details[FIELDS.GENERAL_PARTS] = self.general_segmentator.get_elements_in_image(rgb_uint8_image=rgb_uint8_img,
                                                                                       segmentation_mask=general_segmentation_mask)
        # Include the YOLOv5 details
        details[FIELDS.OBJECT_COUNTING] = self.object_counter.get_details(rgb_uint8_image=rgb_uint8_img)
        simplified_colors_image = self.general_segmentator.get_simplified_colors_segmented_prediction(img=rgb_uint8_img,
                                                                                                      segmentation_mask=general_segmentation_mask,
                                                                                                      with_borders=True)
        # Extract all faces within the image
        rgb_uint8_img = resize(image=rgb_uint8_img, output_shape=general_segmentation_mask.shape,
                               preserve_range=True).astype(np.uint8)
        faces, boxes = self.face_detection.get_cropped_faces(rgb_uint8_image=rgb_uint8_img, margin=0.,
                                                             return_box_position=True)

        # If any face is detected into the image
        if len(faces) > 0:
            # Get the faces with a margin for parsing more information within them
            faces = self.face_detection.get_cropped_faces(rgb_uint8_image=rgb_uint8_img, margin=0.2, boxes=boxes)
            margin_faces, margin_boxes = self.face_detection.get_cropped_faces(rgb_uint8_image=rgb_uint8_img, margin=0.3, boxes=boxes,
                                                                               return_box_position=True)
            details[FIELDS.FACES] = []
            # For each face
            for i, (face, margin_face, margin_box, strict_box) in enumerate(zip(faces, margin_faces, margin_boxes, boxes)):
                x1, y1, x2, y2 = strict_box
                # Estimate its characteristics (apparent age, apparent gender and apparent expression)
                estimations =  {aspect : estimator.predict(face=face) for aspect, estimator
                                                                                in self.face_estimation.items()}
                # Estimate the area that occupies (Area of the face box between the area of the whole image)
                estimations[FIELDS.TOTAL_AREA] = (y2 - y1) * (x2 - x1) / (rgb_uint8_img.shape[0] * rgb_uint8_img.shape[1])
                # Segment the face
                segmented_face_mask = self.face_segmentator.predict(rgb_uint8_image=margin_face)
                simplified_colors_face = self.face_segmentator.get_simplified_colors_segmented_prediction(img=margin_face,
                                                                                                          segmentation_mask=segmented_face_mask,
                                                                                                          with_borders=True, dilate_borders = True)
                simplified_colors_face = resize(image=simplified_colors_face, output_shape=margin_face.shape,
                                preserve_range=True,order=3).astype(np.uint8)
                # Include the details of the face
                estimations[FIELDS.FACE_PARTS] = self.face_segmentator.get_elements_in_image(
                    rgb_uint8_image=margin_face,
                    segmentation_mask=segmented_face_mask)
                # Fuse the face with the general image colored segmentation map, for saving more information about it
                fuse_images_without_background(img=simplified_colors_image, img_to_insert=simplified_colors_face,
                                               img_to_insert_segmentation_mask=segmented_face_mask, box=margin_box)

                details[FIELDS.FACES].append(estimations)
        else:
            # If no faces were detected set this details as Missing Value (None)
            details[FIELDS.FACES] = None

        # Save the segmentation color map if it was requested
        if save_anonymized_img_as is not None:
            dirname = os.path.dirname(save_anonymized_img_as)
            # Create the directory if it did not exist
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            Image.fromarray(simplified_colors_image).save(fp=save_anonymized_img_as)
        return details


