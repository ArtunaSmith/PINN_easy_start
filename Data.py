import numpy as np
import math
import os
import matplotlib.pyplot as plt


class Generator:
    def __init__(self, density, flag, method='gauss'):
        """
        Class Generator
        Return a Generator to generate the 2d data
        :param method: the method control the random process
        :param density: the density of 2d data points which defined as the number of data points per area
        :param flag: the label of the 2d data point,
            which needed to be an integer cause the data here will be used for classification
        """
        self.method = method
        self.density = density
        self.flag = flag

    def line(self, p1: (float, float), p2: (float, float), length: float):
        """
        Generate the data along a line defined with 2 points
        :param p1: first point of the line
        :param p2: second point of the line
        :param length: width of the data point along the line
        :return: (data_feature, data_label), shape(data_feature)=(N, 2), shape(data_label)=(N, 1)
        """

        # Define and calculate some parameters will be used
        x1, x2 = p1[0], p2[0]
        y1, y2 = p1[1], p2[1]
        dx = x2 - x1
        dy = y2 - y1
        distance = (math.sqrt(math.pow(dx, 2) + math.pow(dy, 2)))
        amount = int(distance * length * self.density)
        direction = (dx / distance, dy / distance)
        vertical = (dy / distance, -dx / distance)
        x0, y0 = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        data_f, data_l = np.empty((amount, 2)), np.empty((amount, 1), dtype='int')

        # Generate
        data_l[:, :] = self.flag
        t = np.random.rand(amount) * distance - 0.5 * distance
        data_f[:, 0] = x0 + direction[0] * t
        data_f[:, 1] = y0 + direction[1] * t

        # noise along vertical direction
        noise_f = np.empty((amount, 2))
        noise_f[:, 0] = vertical[0] * 0.5 * length * np.random.randn(amount) * 1.25
        noise_f[:, 1] = vertical[1] * 0.5 * length * np.random.randn(amount) * 1.25
        data_f[:] = data_f[:] + noise_f[:]
        return data_f, data_l

    def rectangle_inside(self, x1: float, x2: float, y1: float, y2: float):
        """
        Generate the data inside a rectangle
        :param x1: the min value of x
        :param x2: the max value of x
        :param y1: the min value of y
        :param y2: the max value of y
        :return: (data_feature, data_label), shape(data_feature)=(N, 2), shape(data_label)=(N, 1)
        """

        # input check
        if x1 == x2 or y1 == y2:
            return None
        if x1 > x2:
            temp = x1
            x1 = x2
            x2 = temp
        if y1 > y2:
            temp = y1
            y1 = y2
            y2 = temp

        # variables definition
        amount = int(abs((x1 - x2) * (y1 - y2)) * self.density)
        output_f, output_l = np.empty((amount, 2)), np.empty((amount, 1), dtype='int')

        # inside points generate
        output_l[:, :] = self.flag
        if self.method == "gauss":
            for i in range(amount):
                while True:
                    x = np.random.randn() * 0.5 * (x2 - x1) / 2.0 + (x1 + x2) / 2.0
                    if x <= x2 and x >= x1:
                        output_f[i, 0] = x
                        break
                while True:
                    y = np.random.randn() * 0.5 * (y2 - y1) / 2.0 + (y1 + y2) / 2.0
                    if y <= y2 and y >= y1:
                        output_f[i, 1] = y
                        break
        if self.method == "uniform":
            output_f[:, 0] = np.random.rand(amount) * (x2 - x1) + x1
            output_f[:, 1] = np.random.rand(amount) * (y2 - y1) + y1
        return output_f, output_l

    def circle_inside(self, center: tuple, radius: float):
        """
        Generate the data inside a circle
        :param center: the center of the circle, tuple with 2 value (x, y)
        :param radius: the radius of circle
        :return: (data_feature, data_label), shape(data_feature)=(N, 2), shape(data_label)=(N, 1)
        """

        # input check
        if radius <= 0:
            return None

        # variables defination
        x0, y0 = center
        amount = int(math.pi * math.pow(radius, 2) * self.density)
        output_f, output_l = np.empty((amount, 2)), np.empty((amount, 1), dtype='int')

        # inside points generate
        output_l[:, :] = self.flag
        if self.method == "gauss":
            for i in range(amount):
                while True:
                    x = np.random.randn() * 0.5 * radius + x0
                    y = np.random.randn() * 0.5 * radius + y0
                    if math.sqrt(math.pow(x - x0, 2) + math.pow(y - y0, 2)) <= radius:
                        output_f[i, 0] = x
                        output_f[i, 1] = y
                        break
        if self.method == "uniform":
            for i in range(amount):
                while True:
                    x = np.random.rand() * 2.0 * radius + x0 - radius
                    y = np.random.rand() * 2.0 * radius + y0 - radius
                    if math.sqrt(math.pow(x - x0, 2) + math.pow(y - y0, 2)) <= radius:
                        output_f[i, 0] = x
                        output_f[i, 1] = y
                        break
        return output_f, output_l

    def circle_outside(self, center: tuple, radius: float, length: float):
        """
        Generate the data outside a circle
        :param center: the center of the circle, tuple with 2 value (x, y)
        :param radius: the radius of circle
        :param length: the width of the distance beyond the radius
        :return: (data_feature, data_label), shape(data_feature)=(N, 2), shape(data_label)=(N, 1)
        """

        # input check
        if radius <= 0:
            return None
        if length <= 0:
            return None

        # variables defination
        x0, y0 = center
        amount = int(math.pi * (math.pow(radius + length, 2) - math.pow(radius, 2)) * self.density)
        output_f, output_l = np.empty((amount, 2)), np.empty((amount, 1), dtype='int')

        # inside points generate
        output_l[:, :] = self.flag
        if self.method == "gauss":
            for i in range(amount):
                while True:
                    theta = np.random.rand() * 2.0 * math.pi
                    distance = abs(np.random.randn()) * 0.5 * length

                    x = x0 + (radius + distance) * math.cos(theta)
                    y = y0 + (radius + distance) * math.sin(theta)

                    if math.sqrt(math.pow(x - x0, 2) + math.pow(y - y0, 2)) >= radius:
                        output_f[i, 0] = x
                        output_f[i, 1] = y
                        break
        if self.method == "uniform":
            for i in range(amount):
                theta = np.random.rand() * 2.0 * math.pi
                distance = np.random.rand() * length
                output_f[i, 0] = x0 + (radius + distance) * math.cos(theta)
                output_f[i, 1] = y0 + (radius + distance) * math.sin(theta)
        return output_f, output_l

    def user_define_data_generate(self):
        """
        User could define a custom data generate function here
        """
        pass


class Plot2d:
    def __init__(self, figsize: tuple, dpi=300, cmap=None, size=2,
                 fname='default.png', title='output image', xlabel='X', ylabel='Y',
                 show=True, save=False):
        """
        Class Plot2d
        Visualization of 2d point and their label
        :param figsize: used in plt.figure()
        :param dpi: used in plt.figure()
        :param cmap: used in plt.figure()
        :param size: the size of point
        :param fname: the filename saved
        :param title: the title of the fig
        :param xlabel: the x label of the fig
        :param ylabel: the y label of the fig
        :param show: whether to show the fig
        :param save:whether to save the fig
        """
        if cmap is None:
            cmap = {0: 'r', 1: 'b', 2: 'g'}
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap
        self.fname = fname
        self.s = size
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.show = show
        self.save = save

    def __call__(self, data: tuple):
        """

        :param data_f: feature of data, with shape (N, 2)
        :param data_l: label of data, with shape (N, 1)
        :param show: whether to show the data image, default is to show (True)
        :param save: whether to save the data image, default is not to save (False)
        :return: None
        """

        # input check
        if (self.save is False) and (self.show is False):
            # Which means no sense, then return None
            return None

        # data to data_f and data_l
        data_f, data_l = data

        # num_l is the num of data's label
        num_l = data_l.max() + 1

        # Plot begin
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        for i in range(num_l):
            equal_i = np.squeeze(data_l == i)
            plt.scatter(
                data_f[:, 0][equal_i],
                data_f[:, 1][equal_i],
                c=self.cmap[i],
                s=self.s
            )

        # check the path, if not exists then create
        if os.path.exists('./img_save') is False:
            os.mkdir('./img_save')

        # Save or Plot
        if (self.save is True) and (self.show is True):
            plt.savefig(f'./img_save/{self.fname}')
            plt.show()
        elif (self.save is False) and (self.show is True):
            plt.show()
        elif (self.save is True) and (self.show is False):
            plt.savefig(f'./img_save/{self.fname}')


class Process:
    """
    Class Process
    It only has static method which could be used to process the 2 or more different data
    """
    @staticmethod
    def join(*datas: tuple) -> tuple:
        """
        Join the 2 different data and shuffle the order of them
        :param datas: data with (feature, label)
        :return: (feature, label) after the joining and shuffling
        """
        # join
        output_f = np.concatenate([datas[i][0] for i in range(len(datas))], 0)
        output_l = np.concatenate([datas[i][1] for i in range(len(datas))], 0)

        # shuffle
        num = output_f.shape[0]
        i_list = np.array(range(num))
        np.random.shuffle(i_list)
        output_f = output_f[i_list, :]
        output_l = output_l[i_list, :]
        return output_f, output_l












