import os
import glob

class FileManagement(object):
    def __init__(self):
        pass

    @staticmethod
    def make_directories(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            print "create folder for", folder