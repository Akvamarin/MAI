"""
IMPORTANT ADVICE:

This specification of the Crawler can only be used for the academic purposes that it was developed
(it was a Master free project (Master on Artificial Intelligence (Universitat Politecnica de Catalunya (UPC))),
of the Big Data Analysis subject).

It does not saves any identifiable information of any user, in accordance with the European GDRP
(General Data Protection Regulation (EU) 2016/679)

Every indirect Instagram identifier is hashed before saving it, and multimedia data is not directly saved.
Instead of it, an extremely simplified segmentation map, which only contains the most characteristic colors of
every region is saved.
"""

from Crawler.SeleniumStandardCrawler import SeleniumStandardCrawler
from Crawler.HashtagsCrawler.Constants import *
from Database import Constants as FIELDS
from Database.Constants import IMAGE_DATASET_PATH, IMAGES_SAVE_FORMAT
from Crawler.commons import *
from numpy.random import shuffle, seed
import os
from random import randrange
from time import sleep
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from warnings import warn
import re
from pymongo.errors import DuplicateKeyError
from time import time
from Database.HashtagDB.HashtagDB import get_ids_of_saved_images
from hashlib import sha256

class HashtagCrawler(SeleniumStandardCrawler):
    """
    Child class which inherits SeleniumStandardCrawler and transform it into a Crawler that crawl through a given Hashtag
    Instagram page. It uses a given feature_extraction_pipeline to anonymize any picture (making it more simple and
    completely not identifiable), and extracting all the relevant information about its content (Objects appearing,
    Number of persons, Gender, Age, Expression, Colors, Areas...). It merges this information with all the
    non identifiable information given by Instagram into the HTML (Date, Other tags, Amount of likes and comments,
    Basic image descriptors...). It saves all this information into the given MongoDB db and the simplified and
    anonymous images that contains the color segmentation into a synchronized dataset.
    """
    def __init__(self, database, feature_extraction_pipeline, base_url=TAGS_WEB, tags_to_explore = TAGS_TO_EXPLORE,
                 chrome_driver_path=CHROME_DRIVER_PATH, dataset_path=IMAGE_DATASET_PATH):
        """
        Child class which implements the concrete Instagram Crawler.
        :param database: MongoDB db instance. Instance of the MongoDB db where save the images.
        :param feature_extraction_pipeline: VisualAnalysisPipeline instance. Instance of the feature extraction pipeline
        that is used for extracting all the information within the images
        :param base_url: String. Base URL of the web page to crawl. In this case the base URL of the
                            Instagram Hashtag search.
        :param tags_to_explore: List of Strings. List of tags to crawl.
        :param chrome_driver_path: ChromeDriver. Path of the ChromeDriver of your installed Chrome Version
        :param dataset_path: String path. Path of the dataset where to save the images that has been previously
                            segmented and simplified for making them totally non identifiables.
        """
        super().__init__(base_url=base_url, chrome_driver_path=chrome_driver_path)
        self.tags_to_explore = tags_to_explore
        # Shuffle the tags that will be visited in order to noise the access pattern
        seed(int(time()))
        shuffle(self.tags_to_explore)

        # Get the ids of the already saved images, for not repeating them
        self.images_explored = get_ids_of_saved_images(dataset_dir=dataset_path)

        self.db = database
        self.feature_extraction_pipeline = feature_extraction_pipeline
        self.consecutive_duplicated_posts = 0
        self.dataset_path = dataset_path

    def collect(self, max_images_to_capture = MAX_IMAGES_TO_CAPTURE_BY_TAG, path_to_save = IMAGE_DATASET_PATH):
        """
        Implements the data mining process which crawls the page, extract the non personal information of every image,
        anonymize it, simplifies it, and save the extracted information into the MongoDB db and the dataset.
        This process is purposely slow, in order to not overwhelm Instagram and being gentle with the process.
        The only requests that this process send to Instagram is the initial request with the tag URL, and some
        simulated scrolls down every several seconds or few minutes.
        :param max_images_to_capture: int. Max amount of images to crawl by every tag. It should be not higher than
        1,000 for not being neither suspicious nor aggressive with the web page.
        :param path_to_save: String path. Path where to save the simplified and anonymous images.
        """
        # For every tag to crawl
        for tag in self.tags_to_explore:
            # Include the tag into the already explored images register
            if not tag in self.images_explored:
                self.images_explored[tag] = {}
            # Acces to the URL of the tag
            url = self.base_url + tag
            self.driver.get(url=url)
            # Generate its directory into the dataset
            path_tag = os.path.join(path_to_save, tag)
            if not os.path.isdir(path_tag):
                os.makedirs(path_tag)
            tag_scrolls = 0
            max_scrolls = randrange(start=MIN_MAX_SCROLS_BY_TAG, stop=MAX_MAX_SCROLLS_BY_TAG, step=1)
            # Keep the track of the images crawled in order to know when to stop
            divs_detected = 0
            while divs_detected < max_images_to_capture:
                # Gets all the divs of the currently visible posts
                image_divs = self.driver.find_elements_by_xpath(xpath=IMAGE_XPATH)[TOP_POST_BY_URL:]
                if len(image_divs) == 0: break
                # For every one of these posts
                for image_div in image_divs:
                    try:
                        # Get the anonymous hash of the post id, for skipping it in the case that it is repeated.
                        image_id = self.get_post_id(div=image_div, anonymous=True)
                        if image_id not in self.images_explored[tag] and image_id != '':
                            # Scroll for localizing the post in the center of the view
                            self.scroll_to_center_of_view(element=image_div)
                            try:
                                # Extract all details about it and save the simplified anonymous image into the dataset.
                                details = self.get_post_details(div=image_div, image_id=image_id, tag=tag)
                                # Insert all the details extracted into the MongoDB
                                self.db.insert(details)
                                # Add this image id to the register
                                self.images_explored[tag][image_id] = True
                                # Update the tracking variables
                                divs_detected += 1
                                self.consecutive_duplicated_posts = 0
                                # First time, when top posts are taken, scroll very down for avoiding those very recent
                                # post for which some Instagram descriptors are not updated yet
                                if divs_detected == TOP_POST_BY_URL:
                                    for i in range(FIRST_SCROLLS):
                                        self.scroll_to_bottom()

                            except NoSuchElementException:
                                # If some problem appeared while extracting the details, erase the image of dataset
                                # (if it was saved) to maintain the consistency.
                                self.delete_image(image_id=image_id, dataset_path=path_tag)
                                warn("Image not saved because mouseover dependent element was lost "
                                     "(Don't move the mouse over the Chrome Window).")
                            except IndexError:
                                # If some problem appeared while converting the id, also erase the image of dataset
                                # (if it was saved) to maintain the consistency.
                                self.delete_image(image_id=image_id, dataset_path=path_tag)
                                warn("Error with the dictionary conversion")
                            except DuplicateKeyError:
                                # If the instance that was inserted already existed keep the track of it.
                                warn("Trying to save an existent instance in the MongoDB collection. Skipped")
                                self.consecutive_duplicated_posts += 1
                                print("Consecutive Duplicates: {dup}".format(dup=self.consecutive_duplicated_posts))
                                if self.consecutive_duplicated_posts > DUPLICATED_ERRORS_THRESHOLD:
                                    break
                        else:
                            print('Image id: {id}'.format(id=image_id))
                            self.consecutive_duplicated_posts += 1
                            print("Consecutive Duplicates increased by 1: {dup}".format(dup=self.consecutive_duplicated_posts))
                            if self.consecutive_duplicated_posts > DUPLICATED_ERRORS_THRESHOLD:
                                break

                    # As Instagram is charged dynamically, some pictures are lost during the scroll
                    except StaleElementReferenceException: pass
                # Do scroll down N times, in order to request for a good set of new photos
                for i in range(int(randrange(MIN_SCROLLS_EVERY_STEP, MAX_SCROLLS_EVERY_STEP))):
                    self.scroll_to_bottom()
                    tag_scrolls += 1
                    if tag_scrolls > max_scrolls:
                        divs_detected = max_images_to_capture
                        print("Search interrupted after {s} scrolls, because of max scrolls where achieved".format(s=tag_scrolls))
                        break
                # Wait a random time for neither overwhelm the page nor be easily detected as a bot.
                sleep(randrange(MIN_SLEEP_TIME, MAX_SLEEP_TIME))
                if self.last_div_ends_the_tag(div=image_divs[-1]):
                    break

    def delete_image(self, image_id, dataset_path = IMAGE_DATASET_PATH, format=IMAGES_SAVE_FORMAT):
        """
        Delete an image from the dataset in the case if it did exist.
        :param image_id: String. ID of the image to delete.
        :param dataset_path: String path. Path of the Image Dataset where the image is located
        :param format: Str Image format. Format in which the image is saved
        """
        path = os.path.join(dataset_path, image_id + '.' + format)
        # If the image exists try to delete it
        if not os.path.isfile(path):
            try:
                os.remove(path)
            except:
                warn("Trying to remove an inexistent image with the id: {id}".format(id=image_id))

    def get_post_id(self, div, anonymous = True):
        """
        Get the post id of the given post div.
        :param div: WebElement. Post div from which to extract the id, which is saved in the parent a attribute
        :param anonymous: Boolean. If anonymize the identifier. If it is intentended to be saved, this value
        must always be True. Default: True.
        :return: String id. Id of the post.
        """
        post_link = div.find_element_by_xpath(xpath=LINK_OF_CURRENT_IMAGE_XPATH).get_attribute(name="href")
        id = post_link[post_link.rfind('p/') + len('p/'):-len('/')]
        if anonymous:
            id = anonymize_id(string_id=id)
        return id

    def get_post_details(self, div, tag, image_id = None):
        """
        Get all the non identifiable details of the post and return them as a Dictionary which is directly parseable to
         json format. Also saves the correspondent image into the synchronized dataset.
        :param div: WebElement. Div containing the post
        :param tag: String. Hashtag by which the post have been found
        :param image_id: String. Id of the image
        :return: Dictionary. Non identifiable details of the post, directly parseable to json.
        """
        # Find the id if it is not given
        if image_id is None:
            image_id = self.get_post_id(div=div,anonymous=True)

        # Get (from a screenshot) the image of the post
        img = self.get_screenshot(element=div)
        # Set the path where to save its non identifiable modification
        img_path = os.path.join(self.dataset_path,tag,image_id+'.'+IMAGES_SAVE_FORMAT) if self.dataset_path is not None\
                                                                                            else None
        # Extract the description of the image
        image_description = div.find_element_by_xpath(DESCRIPTION_OF_CURRENT_IMAGE_XPATH).get_attribute("alt")
        # Simulate the mouse over for extracting the amount of likes and comments
        self.mouse_over_element(element=div)
        # Extract the amount of likes and comments
        likes_and_comments_text = div.find_element_by_xpath(xpath=LIKES_AND_COMMENTS_OF_CURRENT_IMAGE_XPATH).text
        likes_and_comments = [formatted_string_to_int(number=num) for num in likes_and_comments_text.split('\n')]

        likes, comments = likes_and_comments if len(likes_and_comments) == 2 else (None, None)

        # Check if it comes from the Top posts section (True) or from Most recent (False)
        is_highlighted = len(div.find_elements_by_xpath(xpath=CURRENT_IMAGE_IS_HIGHLIGHTED_XPATH)) > 0

        """ # Avoid it for the sake of anymization
        # Check if there is people tagged (in the cases where this information is available from outside)
        people_tagged = re.findall(r'@\w+', image_description)
        people_tagged = [person[len('@'):] for person in people_tagged] if len(people_tagged) > 0 else None
        """

        # Check which other tags have been used (in the cases where this information is available)
        all_tags = re.findall(r'#\w+', image_description)
        all_tags = [tag[len('#'):] for tag in all_tags] if len(all_tags) > 0 else None

        # Check if the post is a Carousel of images
        is_carousel = len(div.find_elements_by_xpath(CURRENT_IMAGE_IS_CAROUSEL_XPATH)) > 0

        # Check if the post is a video
        is_video = len(div.find_elements_by_xpath(CURRENT_IMAGE_IS_VIDEO_XPATH)) > 0 if not is_carousel else False
        # Get the formatted date in which the post was uploaded
        date = extract_date(image_string_descriptor=image_description)

        # Extract the rest of image features, through the vision pipeline
        image_features = self.feature_extraction_pipeline.get_details_of_image(rgb_uint8_img=img,
                                                                             save_anonymized_img_as = img_path)
        # Extract the instagram semantic descriptors and insert them into image descriptors
        image_features[FIELDS.INSTAGRAM_DESCRIPTORS] = extract_semantic_descriptors(image_string_descriptor=image_description)

        # Build the details dictionary, that can be directly parsed to JSON
        details = {FIELDS.ID: image_id, FIELDS.TAG: tag, FIELDS.DATE: date,
                   FIELDS.LIKES: likes, FIELDS.COMMENTS: comments, FIELDS.TOP_POST: is_highlighted,
                   FIELDS.ALL_TAGS: all_tags,
                   FIELDS.IS_CAROUSEL:is_carousel, FIELDS.IS_VIDEO: is_video,
                   FIELDS.IMAGE_FEATURES : image_features}

        return details

    def last_div_ends_the_tag(self, div):
        """
        Check if the last div is empty. It would indicate that all post of the given tag have been already crawled
        :param div: WebElement. Last div of the divs list
        :return: Boolean. True if the last div ends the posts of the tag, False otherwise
        """
        try:
            return len(div.find_elements_by_xpath(CURRENT_IMAGE_IS_LAST_XPATH)) > 0
        except StaleElementReferenceException:
            return False

def extract_semantic_descriptors(image_string_descriptor):
    """
    Parse the descriptor of the image given by Instagram, in order to extract all the elements that they say that the
    image contains.
    :param image_string_descriptor: String. String descriptor of the image given by instagram into the div
    :return: List of String. List with the image descriptors offered by Instagram.
    """
    # Find the descriptors of the Image offered by Instagram
    content = re.search(INSTAGRAM_SEMANTIC_DESCRIPTORS_REGEX, image_string_descriptor)
    # If there are no descriptors, return an empty list
    if content is None: return []
    # If there are descriptors, format them correctly
    content = content.group(0)[len(SEMANTIC_DESCRIPTOR_MARKER):]
    content = re.split(SEMANTIC_SPLIT, content)
    descriptors = []
    # For every descriptor
    for descriptor in content:
        descriptor = descriptor.strip()
        # If it is intended to be translated into another descriptor or into a group, translate it.
        if descriptor in SEMANTIC_TAGS_TRANSLATOR:
            descriptor = SEMANTIC_TAGS_TRANSLATOR[descriptor]
            if descriptor is None: continue
        # Append it to the descriptors
        descriptors.append(descriptor)
    # If no descriptors found set it as Missing Value (None)
    if len(descriptors) == 0:
        descriptors = None
    return descriptors

def extract_date(image_string_descriptor, format = DEFAULT_DATE_FORMAT):
    """
    Parse the post descriptor of the image given by Instagram, in order to extract the uploading date.
    :param image_string_descriptor: String. String descriptor of the image given by instagram into the div
    :param format: String. Format of the date that will be given as output. Default: 'dd-mm-yyyy'
    :return: String. Parsed data into the given format.
    """
    # Find the data
    date = re.search(DATE_REGEX, image_string_descriptor)
    # If there is no date return Missing Value (None)
    if date is None: return None
    # If there was a date extract day, month and year from it
    date = date.group(0)[len(DATE_MARKER):]
    month = MONTH_TO_NUMBER[re.search(MONTH_REGEX, date).group(0)]
    day, year = re.findall(r'\d+', date)
    # Put it into the correct format
    if format == 'dd-mm-yyyy':
        formatted_date = '{:02d}-{:02d}-{:04d}'.format(int(day), int(month), int(year))
    else:
        raise NotImplementedError('Only dd-mm-yyyy format is actually supported')
    return formatted_date




