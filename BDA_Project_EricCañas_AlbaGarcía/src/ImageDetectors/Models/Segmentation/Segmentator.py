from keras_segmentation.data_utils.data_loader import get_image_array
from ImageDetectors.Models.Commons import *
from PIL import Image
import numpy as np
from skimage.transform import resize
from ImageDetectors.Models.Classification.ColorNaming import ColorClassifier, name_to_rgb
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation
from Database import Constants as FIELDS
from ImageDetectors.Models.Constants import MODEL_BY_DATASET, MEANS_BY_DATASET, STD_BY_DATASET


class Segmentator:
    """
    General implementation of a segmentation model, including all the common functions that are useful for the
    concrete crawling purpose. This class is prepared for receiving rgb uint8 images as input and generate as
    output detailed dictionaries and simplified segmentation maps, containing all the esential information within
    the given image.
    """
    def __init__(self, dataset):
        """
        Initializes the model with all the additional attributes that are useful for its usage (The class names read
        from the corresponding CSVs, the pre-process mean and std values for every channel and the 1-NN color naming
        models used for image simplification and object color description.
        :param dataset: String. One of the segmentation dataset CSV names available into the ClassInfo folder.
        """
        # Instantiates the model
        self.model = MODEL_BY_DATASET[dataset]()
        # Set the input size
        _, self.input_h, self.input_w, _ = self.model.input_shape
        # Set the output size
        self.classes = self.model.output_shape[-1]
        # Set the class names that corresponds to every output
        self.class_names = get_class_names(dataset=dataset)
        if len(self.class_names) != self.classes:
            raise RuntimeError("Charged PretrainedModel have {out} output classes, but {dataset} indicates {info}."
                               .format(out=self.classes, dataset=dataset, info=len(self.class_names)))
        # Initialize a fixed RGB color code for every label
        self.color_code = get_color_codes(num_classes=self.classes)
        # Gets an instance of the Color Classifier that is useful for extracting the color name information
        self.color_classifier = ColorClassifier()
        # Get the mean and std of the model, that will be used for pre-processing the input
        self.mean, self.std = MEANS_BY_DATASET[dataset], STD_BY_DATASET[dataset]

    def predict(self, rgb_uint8_image):
        """
        Predict the segmentation map of a given image
        :param rgb_uint8_image: HxWx3 numpy image in uint8 format. Image from which predict the segmentation map
        :return: HxW labels image in int format. Segmentation map of the input image with the identifiers of each
        predicted class as value
        """
        if rgb_uint8_image.shape[-1] != 3:
            raise RuntimeError("Image is not RGB, it has {outchannels} output channels".format(outchannels=rgb_uint8_image.shape[-1]))
        # Pre-process the input if necessary
        if self.mean is None and self.std is None:
            rgb_uint8_image = get_image_array(image_input=rgb_uint8_image, width=self.input_w, height=self.input_h, ordering=None)
        else:
            # Resize it and standardize it with the pre-training means and std of the model
            rgb_uint8_image = resize(rgb_uint8_image, (self.input_w, self.input_h))
            rgb_uint8_image = (rgb_uint8_image - self.mean) / self.std
        # Predict the segmentation map
        prediction = self.model.predict(rgb_uint8_image[None, ...])[0]
        # Convert it into an identifier segmentation map by taking the most probable label for each pixel
        prediction = prediction.reshape((self.input_h, self.input_w, self.classes)).argmax(axis=-1)
        return prediction

    def get_colored_prediction(self, rgb_uint8_image, segmentation_mask = None, with_borders = False,
                               dilate_borders = False, save_as=None):
        """
        Return a colored segmentation map of the rgb_uint8_image
        :param rgb_uint8_image: HxWx3 numpy image in uint8 format. Image from which predict the colored segmentation.
        :param segmentation_mask: HxW segmentation map in int format. If not given, it is predicted. Default: None.
        :param with_borders: Boolean. If True, it borders the segmented objects with a black line. Default: None.
        :param dilate_borders: Boolean. If True, (and with_borders is also True) makes the drawn borders thicker.
                                        Useful when the image will be downscaled later. Default: False.
        :param save_as: str. Path where save the output. If None, the output is not saved, only returned. Default: None.
        :return: HxWx3 numpy uint8 array, where each pixel its colored with the color attached to its class.
        """
        # Predict the segmentation map if it is not given
        if segmentation_mask is None:
            segmentation_mask = self.predict(rgb_uint8_image=rgb_uint8_image)
        # Transform the segmentation map to a colored segmentation map
        colored_img = self.color_code[segmentation_mask]
        # Border the objects within the segmentation map with a black line if requested
        if with_borders:
            colored_img = apply_borders_on_image(rgb_uint8_img=colored_img, segmentation_mask=segmentation_mask,
                                                 dilate_borders=dilate_borders)
        # Save the image into the given path (if given)
        if save_as is not None:
            Image.fromarray(colored_img).save(save_as)
        return colored_img

    def get_simplified_colors_segmented_prediction(self, img, segmentation_mask = None, with_borders = False, dilate_borders=False, save_as=None):
        """
        Return a colored segmentation map of the rgb_uint8_image, where each object is colored with its XKCD
        perceptually nearest median color.
        :param rgb_uint8_image: HxWx3 numpy image in uint8 format. Image from which predict the colored segmentation.
        :param segmentation_mask: HxW segmentation map in int format. If not given, it is predicted. Default: None.
        :param with_borders: Boolean. If True, it borders the segmented objects with a black line. Default: None.
        :param dilate_borders: Boolean. If True, (and with_borders is also True) makes the drawn borders thicker.
                                        Useful when the image will be downscaled later. Default: False.
        :param save_as: str. Path where save the output. If None, the output is not saved, only returned. Default: None.
        :return: HxWx3 numpy uint8 array, where each pixel its colored with the color attached to its class.
        """
        # Predict the segmentation map if it is not given
        if segmentation_mask is None:
            segmentation_mask = self.predict(rgb_uint8_image=img)
        # Resize the image if necessary, in order to make it to match with the segmentation map size
        if img.shape[:2] != segmentation_mask.shape:
            img = resize(image=img, output_shape=segmentation_mask.shape)
        # Get the median color of each object within the image
        classes = np.unique(segmentation_mask)
        _, colors = self.get_class_colors(rgb_uint8_image=img, segmentation_mask=segmentation_mask, classes=classes)
        # Generate the colored image
        colored_img = np.empty(shape=img.shape, dtype = np.uint8)
        for id, color in zip(classes, colors):
            colored_img[segmentation_mask==id] = name_to_rgb(color, return_as_uint8=True)
        # Border the objects within the segmentation map with a black line if requested
        if with_borders:
            colored_img = apply_borders_on_image(rgb_uint8_img=colored_img, segmentation_mask=segmentation_mask,
                                                 dilate_borders=dilate_borders)
        # Save the image into the given path (if given)
        if save_as is not None:
            Image.fromarray(colored_img).save(save_as)
        return colored_img

    def get_elements_in_image(self, rgb_uint8_image, segmentation_mask = None):
        """
        Return a dictionary containing which objects are in the image, which areas they occupy, and which median
        colors they have. Both into the 11 color names space and the XKCD space.
        :param rgb_uint8_image: HxWx3 numpy image in uint8 format. Image from which get the elements within.
        :param segmentation_mask: HxW segmentation map in int format. If not given, it is predicted. Default: None.
        :return: Dictionary. It contains which objects are in the image, which areas they occupy, and which median
        colors they have. Both into the 11 color names space and the XKCD space. The list of each attribute will
        be sorted by the relative area that the object occupy in to the image
        """
        # Predict the segmentation map if it is not given
        if segmentation_mask is None:
            segmentation_mask = self.predict(rgb_uint8_image=rgb_uint8_image)
        # Resize the image if necessary, in order to make it to match with the segmentation map size
        if rgb_uint8_image.shape[:2] != segmentation_mask.shape:
            rgb_uint8_image = resize(image=rgb_uint8_image, output_shape=segmentation_mask.shape)
        # Get the different classes into the segmentation map and the discrete area that they occupy
        classes, areas = np.unique(segmentation_mask, return_counts=True)
        classes_in_image = self.class_names[classes]
        areas = areas/np.sum(areas)
        # Get the color definitions
        basic_11_colors, xkcd_colors = self.get_class_colors(rgb_uint8_image=rgb_uint8_image, segmentation_mask=segmentation_mask,
                                                             classes=classes)
        # Order the information by the area of the object
        sorted_args = np.argsort(areas)[::-1]
        classes_in_image, areas = classes_in_image[sorted_args], areas[sorted_args]
        basic_11_colors, xkcd_colors =  basic_11_colors[sorted_args], xkcd_colors[sorted_args]

        return {FIELDS.PARTS: list(classes_in_image), FIELDS.AREAS: list(areas),
                FIELDS.BASIC_COLORS: list(basic_11_colors), FIELDS.XKCD_COLORS: list(xkcd_colors)}

    def get_class_colors(self, rgb_uint8_image, segmentation_mask, classes):
        """
        Return the median RGB color of each object in classes that appears in the segmentation_mask
        :param rgb_uint8_image: HxWx3 numpy image in uint8 format. Image from which calculate its median object colors.
        :param segmentation_mask: HxW segmentation map in int format. If not given, it is predicted. Default: None.
        :param classes: List of ids. Ids of the for which extract the median color
        :return: Two List of String. One with the 11 basic color name of the median color of each class and another
        with the xkcd color names. Both lists share the order given by classes.
        """
        basic_colors, xkcd_colors = [], []
        # For each object
        for id in classes:
            # Calculate the median RGB color
            median_rgb_color = np.median(rgb_uint8_image[segmentation_mask == id], axis=0)
            # Extract the 11 basic color name
            basic_colors.append(self.color_classifier.get_basic_color_name(rgb_color=median_rgb_color))
            # Extract the xkcd color name
            xkcd_colors.append(self.color_classifier.get_xkcd_color_name(rgb_color=median_rgb_color))
        return np.array(basic_colors, dtype='<U32'), np.array(xkcd_colors, dtype='<U64')

def apply_borders_on_image(rgb_uint8_img, segmentation_mask, border_color = (0, 0, 0), dilate_borders=False):
    """
    Draw over an image the borders that surround the objects determined by a given segmentation_mask.

    :param rgb_uint8_image: HxWx3 numpy image in uint8 format. Image to which draw the borders
    :param segmentation_mask: HxW segmentation map in int format. Segmentation map containing the identifier of the
    objects that will be bordered.
    :param border_color: Three uint8 values tuple or list. Color of the borders to draw
    :param dilate_borders: Boolean. If True, make the borders thicker. Default: False.
    :return: HxWx3 numpy image in uint8 format. rgb_uint8_img with the borders drawn according to the segmentation map.
    """
    # Define the borders
    borders = find_boundaries(label_img=segmentation_mask)
    # Dilate them if necessary
    if dilate_borders:
        borders = binary_dilation(borders)
    # Draw them over the given image
    rgb_uint8_img[borders] = border_color
    return rgb_uint8_img