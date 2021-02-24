import cv2
import numpy
import math


RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NUM_TOTAL_POINTS = 68


class ImageObject:
    """docstring for ImageObject"""
    def __init__(self, img, landmarks=None, type="face", sub_type=None):
        self.data = img

        # facial landmarks
        self.landmarks = landmarks

        self.type = type
        self.sub_type = sub_type

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def get_landmarks(self):
        return self.landmarks

    def set_landmarks(self, landmarks):
        self.landmarks = landmarks

    def crop(self, margin, ratio, target_width):

        img = self.data.astype(numpy.float64)
        target_height = target_width / ratio

        # get cropped rect and cropped landmarks
        if self.has_face():
            cropped_rect = self.__get_cropped_rect(margin, ratio)
            self.__get_cropped_landmarks(cropped_rect, target_width, target_height)

        else:
            cropped_rect = ((0, 0), (self.data.shape[1], self.data.shape[0]))

        # get cropped and resized face image.
        [(cropped_left, cropped_top), (cropped_right, cropped_bottom)] = cropped_rect
        width = cropped_right - cropped_left
        height = cropped_bottom - cropped_top

        mat = numpy.float32([[1, 0, -cropped_left], [0, 1, -cropped_top]])
        crop = cv2.warpAffine(img, mat[:2], (width, height))
        crop = crop.astype(numpy.uint8)

        crop = cv2.resize(crop, (int(target_width), int(target_height)))

        self.data = crop

    def align(self):

        if self.has_face() is False:
            return
        origin = self.data.copy()

        # get the face pose infomation
        roll, _ = self.get_face_pose()

        # rotate the image by roll degree
        (h, w) = origin.shape[:2]
        (cen_x, cen_y) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cen_x, cen_y), -roll, 1.0)

        cos = numpy.abs(M[0, 0])
        sin = numpy.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (new_w / 2) - cen_x
        M[1, 2] += (new_h / 2) - cen_y

        self.data = cv2.warpAffine(origin, M, (new_w, new_h))

        # recalculate the landmarks by rotating degree roll
        for i in range(len(self.landmarks)):
            p = (self.landmarks[i][0] - cen_x, self.landmarks[i][1] - cen_y)
            rot_x = math.cos(math.radians(roll)) * p[0] - math.sin(math.radians(roll)) * p[1] + new_w/2
            rot_y = math.sin(math.radians(roll)) * p[0] + math.cos(math.radians(roll)) * p[1] + new_h/2
            self.landmarks[i] = [int(rot_x), int(rot_y)]

    def has_face(self):
        # returns true if image has landmarks
        if self.landmarks is not None and len(self.landmarks) == NUM_TOTAL_POINTS:
            return True
        else:
            return False

    def get_face_pose(self):

        if self.has_face() is False:
            return

        # Init the Point of interest
        # Calculate the roll
        right_eye = [0.0, 0.0]
        left_eye = [0.0, 0.0]

        # Eye: Fine the center of the right eye by averaging the points
        for i in RIGHT_EYE_POINTS:
            right_eye[0] += self.landmarks[i][0] / len(RIGHT_EYE_POINTS)
            right_eye[1] += self.landmarks[i][1] / len(RIGHT_EYE_POINTS)
        # Find the center of the left eye by averaging the points
        for i in LEFT_EYE_POINTS:
            left_eye[0] += self.landmarks[i][0] / len(LEFT_EYE_POINTS)
            left_eye[1] += self.landmarks[i][1] / len(LEFT_EYE_POINTS)
        center_eye = [(right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2]

        # Calculate the roll
        roll = - math.atan((left_eye[1] - right_eye[1]) / (left_eye[0] - right_eye[0])) * 180 / math.pi

        # Mouth: left corner and right corner
        left_mouth = [(self.landmarks[54][0] + self.landmarks[64][0]) / 2,
                      (self.landmarks[54][1] + self.landmarks[64][1]) / 2]
        right_mouth = [(self.landmarks[48][0] + self.landmarks[60][0]) / 2,
                       (self.landmarks[48][1] + self.landmarks[60][1]) / 2]
        # Nose:
        center_nose = self.landmarks[33]

        # Jaw:
        center_jaw = self.landmarks[8]

        # Calculate the yaw
        yaw = 0 * 180 / math.pi
        res_yaw = self.__vector_point_to_line(center_nose, (center_jaw, center_eye))

        return roll, res_yaw

    def __get_cropped_landmarks(self, rect, target_width, target_height):

        [(left, top), (right, bottom)] = rect
        points = []
        width = right - left
        height = bottom - top

        for p in self.landmarks:
            p_x = (p[0] - left) * target_width / width
            p_y = (p[1] - top) * target_height / height
            points.append([int(p_x), int(p_y)])

        self.landmarks = points

    def __get_cropped_rect(self, margin, ratio):

        # returns the cropped version of the image
        points = list(zip(*self.landmarks))

        (left, top, right, bottom) = (min(points[0]), min(points[1]), max(points[0]), max(points[1]))
        # (left, top, right, bottom) = (self.landmarks[36][0], min(points[1]), self.landmarks[45][0], max(points[1]))

        width = right - left
        height = bottom - top

        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        new_left = int(center_x - width * (1 + margin) / 2)
        new_right = int(center_x + width * (1 + margin) / 2)
        new_top = int(center_y - height * (1 + margin) / 2 / ratio)
        new_bottom = int(center_y + height * (1 + margin) / 2 / ratio)

        return [(new_left, new_top), (new_right, new_bottom)]

    # Calculate the vector from p to line AB
    def __vector_point_to_line(self, p, line):
        """
        point: p   ,    line: AB
        distance(p, AB) = (-(p.x-A.x)*(B.y-A.y) + (p.y-A.y)*(B.x-A.x))/ length(AB)

            directional vector of line AB : (A[0]-B[0], A[1]-B[1])
            Normal vector : ( A[1]-B[1], -(A[0]-B[0]) )

        point M(x,y) on line AB can be described like this:
            (y-A.y)/(x-A.x) = (B.y-A.y)/(B.x-A.x)
            y = (B.y-A.y)/(B.x-A.x) * (x-A.x) + A.y

            if y > 0:   M is above than line AB
            else:       M is below that line AB

        result vector : (vec_x, vec_y)
        """
        A = line[0]
        B = line[1]
        dis = ((-1) * (p[0] - A[0]) * (B[1] - A[1]) + (p[1] - A[1]) * (B[0] - A[0])) / self.__distance_points(A, B)

        if B[0] - A[0] != 0:
            y = (B[1] - A[1]) / (B[0] - A[0]) * (p[0] - A[0]) + A[1]
            if y < 0:  # Point p is below than line AB
                vec_x = -(A[1] - B[1])
                vec_y = (A[0] - B[0])
            else:  # Point p is above than line AB
                vec_x = (A[1] - B[1])
                vec_y = -(A[0] - B[0])

                # Normalize the directional vector and then multiply with distance p and line_AB
                vec_x = vec_x * dis / math.sqrt(vec_x ** 2 + vec_y ** 2)
                vec_y = vec_y * dis / math.sqrt(vec_x ** 2 + vec_y ** 2)

        else:
            vec_x = (p[0] - B[0])
            vec_y = 0

        return vec_x, vec_y

    @staticmethod
    # Get distance between two points
    def __distance_points(point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
