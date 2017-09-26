# The class is to generate simple training/testing data set for deep learning algorithm study
import os
import numpy as np
import pickle
import random
from toolbox.file_management import FileManagement

class DataGenerator(object):
    def __init__(self, output, kind='dot', number=500, scale=(32, 32)):
        """
        set up parameters for data generation
        """
        super(DataGenerator, self).__init__()
        self.output = output
        self.kind = kind
        self.number = number
        self.scale = scale

    def generate_dot_images_set(self):
        """
        generate dot images and save them with labels in a pickle file.
        """
        data = {}
        for i in range(self.number):
            x = random.randint(0, self.scale[0] - 1)
            y = random.randint(0, self.scale[1] - 1)
            if (x, y) in data:
                continue
            canvas = np.zeros(self.scale)
            canvas[x, y] = 1.0
            data[(x,y)] = canvas
        res = [[data[x], x] for x in data]
        FileManagement.make_directories(os.path.dirname(self.output))
        pickle.dump(res, open(self.output, 'wb'))

    def generate_data(self):
        """
        entry point to generate different types of data
        """
        if self.kind == 'dot':
            self.generate_dot_images_set()