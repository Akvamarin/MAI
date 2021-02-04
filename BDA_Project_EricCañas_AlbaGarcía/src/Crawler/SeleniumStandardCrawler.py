from selenium import webdriver
from Crawler.HashtagsCrawler.Constants import *
from Crawler.GenerateCookies import generate_and_save_cookies
from time import sleep
from random import uniform
import pickle
import re
import sys
import io
from PIL import Image
import numpy as np
from warnings import warn
from selenium.webdriver.common.action_chains import ActionChains

class SeleniumStandardCrawler:
    """
    Parent generic class which initializes the webdriver with the saved cookies and in a inconspicuous way.
    Also includes all the generic methods that could be useful for any crawling scheme.
    """
    def __init__(self, base_url, chrome_driver_path=CHROME_DRIVER_PATH):
        """
        Initializes the Selenium Webdriver setting the preferred language and disabling all the automation flags,
        loading the get base_url web page and re-loading the last saved cookies (or asking for them if not found).
        :param base_url: String. Url of the base web page to crawl.
        :param chrome_driver_path: Path of the Selenium Chrome driver (Must match with the Chrome Version installed).
        """
        # Select the preferred languages and other options for not making obvious that you're an automatic tool
        options = webdriver.ChromeOptions()
        # Set the preferred language, for making the process both inconspicuous and consistent
        options.add_experimental_option('prefs', {'intl.accept_languages': ','.join(PREFERRED_LANGUAGES)})
        # Exclude all the obvious automating flags
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("--disable-blink-features=AutomationControlled")
        # Make the windows to start maximized in order to not maintain the default suspicious resolution of Selenium.
        options.add_argument("--start-maximized")

        # Initialize the driver
        self.driver = webdriver.Chrome(executable_path=chrome_driver_path, chrome_options=options)
        # Reset the size to half of the screen for comfort
        w, h = self.driver.get_window_size().values()
        self.driver.set_window_rect(x=0, y=0, width=w//2+int(np.random.uniform(-10, 10)), height=h)
        # Base URL of the web to Crawl
        self.base_url = base_url
        # Base URL of the domain
        self.domain_url = base_url[:re.search(ALL_DOMAINS_REGEX, base_url).regs[0][-1]]
        # Get the base webpage for charging its cookies
        self.driver.get(self.domain_url)
        # Charge the cookies with the token session if they are give
        try:
            _ = [self.driver.add_cookie(cookie) for cookie in pickle.load(open(COOKIES_FILE_PATH, 'r'+PICKLE_FORMAT))]
        except:
            # If cookies did not exist, generate and save them.
            warn("Cookies are not valid, generate them again")
            sys.stderr.flush(), sys.stdout.flush()
            generate_and_save_cookies(driver=self.driver)

    def scroll_to_center_of_view(self, element, sleep_min_time=0.2, sleep_max_time=0.4):
        """
        Move the scroll in order to place the given element in the center of the view.
        :param element: WebElement. WebElement to place in the center of the screen
        :param sleep_min_time: float. Minimum seconds to sleep after performing the action
        :param sleep_min_time: float. Maximum seconds to sleep after performing the action
        """
        window_size = self.driver.get_window_size()
        self.driver.execute_script("window.scroll(" + str(window_size.get('height')) + ", " +
                                   str(element.location['y'] - window_size.get('width') / 2) + ");")

        sleep(uniform(sleep_min_time, sleep_max_time))

    def scroll_to_bottom(self, sleep_min_time=0.5, sleep_max_time=1.):
        """
        Scroll down to the bottom of page.
        """
        self.driver.execute_script("window.scroll(0, document.body.scrollHeight);")
        sleep(uniform(sleep_min_time, sleep_max_time))

    def mouse_over_element(self, element):
        """
        Simulate that the mouse pass over the given element, in order to activate any elements depending on this action.
        :param element: WebElement. WebElement which mouse over action will be activated
        """
        ActionChains(self.driver).move_to_element(element).perform()

    def get_screenshot(self, element):
        """
        Returns a RGB screenshot of the given element as numpy array in format uint8
        :param element: WebElement. WebElement of which take a screenshot
        :return: Numpy Array in format HxWx3, uint8. RGB screenshot of the element
        """
        with io.BytesIO(element.screenshot_as_png) as img_buffer:
            img = Image.open(img_buffer)
            img = np.array(img.convert(mode='RGB'), dtype=np.uint8)
        return img

    def close(self):
        self.driver.quit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
