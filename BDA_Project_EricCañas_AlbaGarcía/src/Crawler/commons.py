from hashlib import sha256


MULTIPLIERS = {'K':10**3, 'k':10**3, 'M':10**6, 'm':10**6}
DEFAULT_DATE_FORMAT = 'dd-mm-yyyy'
HASH_LEN = 12

def formatted_string_to_int(number):
    """
    Transform an string number in format like: 2.1K, or 2,100 in an integer
    :param number: String. Number in a string format like 2.1K, or 2,100.
    :return: Integer. number casted to int
    """
    # Replace the thousands separator with nothing
    number = number.replace(',','')
    # Get the multiplier which corresponds to the letter or 1 if there is no known letter
    multiplier = MULTIPLIERS.get(number[-1], 1)
    if multiplier != 1:
        number = number[:-1]
    return int(float(number)*multiplier)

def anonymize_id(string_id, hash_len=HASH_LEN):
    """
    Anonymize a given string identificator in order to break its relation with any indirectly identifiable information.
    :param string_id: String. Identificator to anonymize
    :param hash_len: Int. Max lenght of the output hash
    :return: Str. Anonymized id, hashed by sha256 algorithm.
    """
    return sha256(bytes(string_id, encoding='utf-8')).hexdigest()[:hash_len]