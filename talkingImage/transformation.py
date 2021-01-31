import numpy as np
import cv2


class LipsTransform:
    def __init__(self, lf):
        self.lf = lf
        self.img = np.copy(lf.img)

    @staticmethod
    def get_distance(lf):
        """Euclidean distance between 9 and the end of image."""
        vct = (int)((lf.points[8][1] - lf.points[57][1]) / 2)
        return vct

    @staticmethod
    def inPolygon(x, y, xp, yp):
        c = 0
        for i in range(len(xp)):
            if ((yp[i] <= y < yp[i - 1]) or (yp[i - 1] <= y < yp[i])) and (
                    x > (xp[i - 1] - xp[i]) * (y - yp[i]) / (yp[i - 1] - yp[i]) + xp[i]): c = 1 - c
        return c

    @staticmethod
    def findInternalPoints(start_point, end_point, x_points, y_points):
        internal_points = []
        for i in range(start_point[0], end_point[0]):
            for j in range(start_point[1], end_point[1]):
                if LipsTransform.inPolygon(i, j, x_points, y_points) == 1:
                    internal_points.append([i, j])
        return np.array(internal_points)

    @staticmethod
    def replace_points(points, shift, image, new_image, direction=True):
        """Replace points a certain distance.
           Begin points make black"""
        for point in points:
            new_image[point[1]][point[0]] = [0, 0, 0]
        for point in points:
            if direction and (point[1] - shift) < 0:
                new_image[point[1] - shift][point[0]] = image[point[1]][point[0]]  # need exception
            elif (point[1] + shift) < new_image.shape[0]:
                new_image[point[1] + shift][point[0]] = image[point[1]][point[0]]  # need exception
        return new_image

    def transformate(self, coeff):
        n_chin_points = [6, 48, 60, 67, 66, 65, 64, 54, 10, 9, 8, 7]
        chin_points = []
        for n in n_chin_points:
            chin_points.append(self.lf.points[n])
        chin_points = np.array(chin_points)
        start_point = np.array([self.lf.points[48][0], self.lf.points[50][1]])
        end_point = np.array([self.lf.points[54][0], self.lf.points[8][1]])
        inner_chin_points = LipsTransform.findInternalPoints(start_point, end_point,
                                                             chin_points[:, 0], chin_points[:, 1])
        shift = int(LipsTransform.get_distance(self.lf) * coeff)

        new_img = np.copy(self.lf.img)
        new_img = LipsTransform.replace_points(inner_chin_points, shift, self.img, np.copy(new_img), False)

        return new_img

    def plot_area(self, n_points, image):
        prev_point = n_points[0]
        line_thickness = 2
        for i in n_points:
            cv2.line(image, (self.lf.points[prev_point][0], self.lf.points[prev_point][1]),
                     (self.lf.points[i][0], self.lf.points[i][1]), (0, 255, 0), line_thickness)
            prev_point = i
        cv2.line(image, (self.lf.points[prev_point][0], self.lf.points[prev_point][1]),
                 (self.lf.points[n_points[0]][0], self.lf.points[n_points[0]][1]),
                 (0, 255, 0), line_thickness)
        return image
