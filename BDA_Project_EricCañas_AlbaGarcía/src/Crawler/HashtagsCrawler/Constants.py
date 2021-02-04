import os

from ImageDetectors.Models.Classification.EmotionClassifier import EmotionClassifier
from ImageDetectors.Models.Detection.FaceDetector import OpenCVFaceDetector
from ImageDetectors.Models.Estimation.AgeEstimator import AgeEstimator
from ImageDetectors.Models.Classification.GenderClassifier import GenderEstimator
from Database import Constants as FIELDS

CHROME_DRIVER_PATH = os.path.join('Crawler','Drivers','chromedriver87.exe')
COOKIES_FILE_PATH = os.path.join('Crawler', 'Cookies', 'cookies.pkl')

PICKLE_FORMAT = 'b'
ALL_DOMAINS_REGEX = r'.com|.net|.org|.biz|.edu|.info|.es|.ly|.io'
PREFERRED_LANGUAGES = ['en','en_US', 'en_GB']

BASE_WEB = "https://www.instagram.com/explore/"
TAGS_WEB = BASE_WEB + "tags/"
TAGS_TO_EXPLORE = ['pizza', 'luxury', 'christmas', 'pretty', 'makeup', 'kawaii', 'beautiful', 'handsome', 'girl', 'cute', 'boy', 'me']
TOP_POST_BY_URL = 9
FIRST_SCROLLS = 5
MAX_IMAGES_TO_CAPTURE_BY_TAG = 1200/len(TAGS_TO_EXPLORE)
MIN_MAX_SCROLS_BY_TAG, MAX_MAX_SCROLLS_BY_TAG = 28, 36
DUPLICATED_ERRORS_THRESHOLD = 16


MIN_SLEEP_TIME, MAX_SLEEP_TIME = 5., 8.
MIN_SCROLLS_EVERY_STEP, MAX_SCROLLS_EVERY_STEP = 5, 8
# Internal values
DECIMALS = 2

LIKES_COMMENTS_UL_TAG = 'Ln-UN'

# CLASS CODES OF THE WEB
IMAGE_DIV_CLASS = 'eLAPa'
MORE_THAN_IMAGE_CLASS = 'u7YqG'
HIGHLIGHTED_POSTS_CLASS = 'EZdmt'
EMPTY_PLACE_CLASS = '_bz0w'

CAROUSEL_SPAN_ARIA_LABEL = 'Carousel'
VIDEO_SPAN_ARIA_LABEL = 'Video'

LIKES_SPAN_CLASS = 'coreSpriteHeartSmall'
COMMENTS_SPAN_CLASS = 'coreSpriteSpeechBubbleSmall'

# XPATHS TO PARSE
IMAGE_XPATH = './/div[@class="{divtag}"]'.format(divtag=IMAGE_DIV_CLASS)
LINK_OF_CURRENT_IMAGE_XPATH = './parent::a[@href]'
DESCRIPTION_OF_CURRENT_IMAGE_XPATH = ".//img[@alt]"
LIKES_AND_COMMENTS_OF_CURRENT_IMAGE_XPATH = './following-sibling::div/ul[@class = "{likecommentsul}"]'.format(likecommentsul=LIKES_COMMENTS_UL_TAG)
CURRENT_IMAGE_IS_HIGHLIGHTED_XPATH = './ancestor::div[@class="{highlighted}"]'.format(highlighted=HIGHLIGHTED_POSTS_CLASS)
CURRENT_IMAGE_IS_CAROUSEL_XPATH = './following-sibling::div[@class="{more}"]/span[@aria-label="{carousel}"]'.\
                                    format(more=MORE_THAN_IMAGE_CLASS, carousel=CAROUSEL_SPAN_ARIA_LABEL)
CURRENT_IMAGE_IS_VIDEO_XPATH = './following-sibling::div[@class="{more}"]/span[@aria-label="{video}"]'.\
                                    format(more=MORE_THAN_IMAGE_CLASS, video=VIDEO_SPAN_ARIA_LABEL)
CURRENT_IMAGE_IS_LAST_XPATH = './parent::a[@href]/parent::div/following-sibling::div[@class="{finalpost}"]'.format(finalpost=EMPTY_PLACE_CLASS)

SEMANTIC_DESCRIPTOR_MARKER = 'Image may contain: '
SEMANTIC_SPLIT = ', | and '
INSTAGRAM_SEMANTIC_DESCRIPTORS_REGEX = r'{marker}[\d+|\w+| |{split}]*'.format(marker=SEMANTIC_DESCRIPTOR_MARKER, split=SEMANTIC_SPLIT)
DATE_MARKER = 'on '
MONTH_TO_NUMBER = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
MONTH_REGEX = r'({allmonths})'.format(allmonths='|'.join(list(MONTH_TO_NUMBER.keys())))
DATE_REGEX = r''+DATE_MARKER+MONTH_REGEX+' [0-9]{1,2}, \d{4}'
SEMANTIC_TAGS_TRANSLATOR = {'one or more people': 'people', '1 person' : 'person', 'text that says': 'text'}


GENERAL_SEGMENTATION_SETS = ['ADE20K', 'VOC12']
FACE_SEGMENTATION_SET = 'FACE_PARSING'
FACE_ESTIMATOR_MODELS = {FIELDS.GENDER : GenderEstimator, FIELDS.AGE : AgeEstimator, FIELDS.EMOTION : EmotionClassifier}
FACE_DETECTOR = OpenCVFaceDetector

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36"