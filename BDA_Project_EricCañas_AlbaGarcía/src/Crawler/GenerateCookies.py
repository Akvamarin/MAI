import pickle
from Crawler.HashtagsCrawler.Constants import COOKIES_FILE_PATH, PICKLE_FORMAT

def generate_and_save_cookies(driver, save_at = COOKIES_FILE_PATH):
    """
    Waits until the user have accepted the cookies and login into the web for saving those cookies. In this way,
    the following times that the crawler will be launched, they (including any preference and the Authentication Token)
    will be re-loaded, thus avoiding any login step and avoiding suspicious multiple login.
    :param driver: WebDriver. Instance of the Selenium Driver that has opened the web.
    :param save_at: Str path. Path where save the cookies for loading them in following connections.
    """
    # Wait until the user will have generated all cookies
    _ = input("Write your credentials on the explorer and press Enter once you're logged")
    # Save them
    pickle.dump(driver.get_cookies(), open(save_at, 'w'+PICKLE_FORMAT))
