import cv2
import dlib
import numpy


JAW_POINTS = list(range(0, 17))
FACE_POINTS = list(range(17, 68))

RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 60))
LIP_POINTS = list(range(60, 68))

ALIGN_POINTS = (LEFT_BROW_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + RIGHT_EYE_POINTS + MOUTH_POINTS + NOSE_POINTS +
                JAW_POINTS)


class LandmarkDetector:
    def __init__(self, predictor_path=None):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def warp_image(self, im, M, dshape, rect):

        output = numpy.zeros(dshape, dtype=im.dtype)

        # For only rotation of image, normalize the affine coefficients of x and y
        x_ratio = numpy.sqrt(M[0][0, 0] ** 2 + M[0][0, 1] ** 2)
        y_ratio = numpy.sqrt(M[1][0, 0] ** 2 + M[1][0, 1] ** 2)
        M[0] = M[0] / x_ratio
        M[1] = M[1] / y_ratio

        # Transfer the detected the face rect so that, rect can be centered on the new image.
        # Calculate the pre center position of original image.
        face_cen_x = (rect[0][0] + rect[1][0]) / 2
        face_cen_y = (rect[0][1] + rect[1][1]) / 2

        # Calculate the offset for transfer to be centered
        new_cen_x = (dshape[1] / 2) * M[0][0, 0] + (dshape[0] / 2) * M[0][0, 1]
        new_cen_y = (dshape[1] / 2) * M[1][0, 0] + (dshape[0] / 2) * M[1][0, 1]

        # Transfer with 0th coefficients of M matrix
        M[0][0, 2] = face_cen_x - new_cen_x
        M[1][0, 2] = face_cen_y - new_cen_y

        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output


    def transformation_from_points(self, points1, points2):
        """
        Return an affine transformation [s * R | T] such that:
            sum ||s*R*p1,i + T - p2,i||^2
        is minimized.
        """
        # Solve the procrustes problem by subtracting centroids, scaling by the
        # standard deviation, and then using the SVD to calculate the rotation.

        points1 = numpy.matrix(points1)
        points2 = numpy.matrix(points2)
        points1 = points1.astype(numpy.float64)
        points2 = points2.astype(numpy.float64)

        c1 = numpy.mean(points1, axis=0)
        c2 = numpy.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = numpy.std(points1)
        s2 = numpy.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = numpy.linalg.svd(points1.T * points2)

        # The R we seek is in fact the transpose of the one given by U * Vt. This
        # is because the above formulation assumes the matrix goes on the right
        # (with row vectors) where as our solution requires the matrix to be on the
        # left (with column vectors).
        R = (U * Vt).T

        return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                           c2.T - (s2 / s1) * R * c1.T)),
                             numpy.matrix([0., 0., 1.])])


    # Crop the face from the single image (for the single image processing such as emotion detection)
    def detect(self, img):

        # Detect the face rects
        upsample = 0
        dets = self.detector(img, upsample)
        if len(dets) == 0:
            print("    No face found... Upsampling...")
            upsample += 1
            dets = self.detector(img, upsample)
            if len(dets) == 0:
                print("    Also No face found, Even Upsampled")
                return None

        # Get the face rect with maximum size, if no face then return None
        max_det = dets[0]
        for det in dets:
            if det.area() > max_det.area():
                max_det = det

        points = []
        for p in self.predictor(img, max_det).parts():
            points.append([p.x, p.y])

        return points
