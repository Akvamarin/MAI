from pymongo import MongoClient
from Database.Constants import *


class MongoDB:
    """
    General Implementation of the MongoDB connection containing all the functions that could be useful for a general
    crawling project
    """
    def __init__(self, collection_name, host=DEFAULT_HOST, port=DEFAULT_PORT):
        """
        General implementation of the MongoDB connection used for crawling
        :param collection_name: String. Name of the collection where to work
        :param host: String. Ip of the host where the db is located
        :param port: Int. Port from where the db service is offered
        """
        self.client = MongoClient(host=host, port=port)
        self.db = self.client[collection_name]

    def close(self):
        self.client.close()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Override of the exit function for closing the client connection on exit
        """
        self.close()


