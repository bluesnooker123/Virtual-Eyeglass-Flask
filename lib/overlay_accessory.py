"""--------------------------------------------------------------------------------------
                               FACE OVERLAY WITH OBJECTS

distance_points

overlay_glasses
overlay_mustache
overlay_flowercrown
overlay_hat

overlay_accessory(main)
--------------------------------------------------------------------------------------"""

import cv2
import math
import numpy

JAW_POINTS = list(range(0, 17))
FACE_POINTS = list(range(17, 68))

RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 61))
LIP_POINTS = list(range(61, 68))


# Get distance between two points
def distance_points(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def overlay_glasses(src_obj, acc_obj):
    landmarks = src_obj.get_landmarks()
    img = src_obj.get_data()
    acc_info = acc_obj.get_landmarks()
    acc_img = acc_obj.get_data()

    # Calculate the eyes location
    right_eye = [0.0, 0.0]
    left_eye = [0.0, 0.0]

    for i in RIGHT_EYE_POINTS:
        right_eye[0] += landmarks[i][0] / len(RIGHT_EYE_POINTS)
        right_eye[1] += landmarks[i][1] / len(RIGHT_EYE_POINTS)
    for i in LEFT_EYE_POINTS:
        left_eye[0] += landmarks[i][0] / len(LEFT_EYE_POINTS)
        left_eye[1] += landmarks[i][1] / len(LEFT_EYE_POINTS)

    # Calculate the scale ratio
    scale = distance_points(left_eye, right_eye) / distance_points([*acc_info['left_center'].values()],
                                                                   [*acc_info['right_center'].values()])
    center_img = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2)
    center_obj = (acc_info['center']['x'], acc_info['center']['y'])

    roll, pose_vect = src_obj.get_face_pose()
    M = cv2.getRotationMatrix2D(center_obj, roll, scale)

    new_center = [0.0, 0.0]
    new_center[0] = center_obj[0] * M[0][0] + center_obj[1] * M[0][1]
    new_center[1] = center_obj[0] * M[1][0] + center_obj[1] * M[1][1]
    M[0][2] = center_img[0] - new_center[0] + pose_vect[0] * 2 / 3
    M[1][2] = center_img[1] - new_center[1] + pose_vect[1] * 2 / 3

    warp = cv2.warpAffine(acc_img, M,
                          (img.shape[1], img.shape[0]))  # Affine transform and then Convert png image to RGB image

    # Get the mask image with alpha channel of object image
    warp_mask = cv2.warpAffine(acc_img[:, :, 3], M, (img.shape[1], img.shape[0]))

    overlay = numpy.zeros(img.shape)
    for c in range(0, 3):
        overlay[:, :, c] = (img[:, :, c] * (1.0 - (warp_mask[:, :] / 255.0))) + warp[:, :, c] * (
                    warp_mask[:, :] / 255.0)

    # Covert float32 numpy array to uint8
    overlay = overlay.astype(numpy.uint8)

    return overlay


def overlay_mustache(src_obj, acc_obj):
    # Calculate the mustache location
    # Used Landmarks lip points : 60,64,62,    nose: 33
    landmarks = src_obj.get_landmarks()
    img = src_obj.get_data()
    acc_info = acc_obj.get_landmarks()
    acc_img = acc_obj.get_data()

    # Calculate the scale ratio
    scale = distance_points(landmarks[64], landmarks[60]) / distance_points(acc_info[2], acc_info[3])

    center_mustache_img = ((landmarks[62][0] + landmarks[33][0]) / 2, (landmarks[33][1] + landmarks[62][1]) / 2)
    center_mustache_obj = (acc_info[1][0], acc_info[1][1])

    roll, pose_vect = src_obj.get_face_pose()
    M = cv2.getRotationMatrix2D(center_mustache_obj, roll, scale)

    new_center = [0.0, 0.0]
    new_center[0] = center_mustache_obj[0] * M[0][0] + center_mustache_obj[1] * M[0][1]
    new_center[1] = center_mustache_obj[0] * M[1][0] + center_mustache_obj[1] * M[1][1]
    M[0][2] = center_mustache_img[0] - new_center[0]
    M[1][2] = center_mustache_img[1] - new_center[1]

    # acc_img = acc_info[0]

    warp = cv2.warpAffine(acc_img, M,
                          (img.shape[1], img.shape[0]))  # Affine transform and then Convert png image to RGB image

    # Get the mask image with alpha channel of object image
    warp_mask = cv2.warpAffine(acc_img[:, :, 3], M, (img.shape[1], img.shape[0]))

    overlay = numpy.zeros(img.shape)
    for c in range(0, 3):
        overlay[:, :, c] = (img[:, :, c] * (1.0 - (warp_mask[:, :] / 255.0))) + warp[:, :, c] * (
                    warp_mask[:, :] / 255.0)

    # Covert float32 numpy array to uint8
    overlay = overlay.astype(numpy.uint8)

    return overlay


def overlay_flowercrown(src_obj, acc_obj):
    landmarks = src_obj.get_landmarks()
    img = src_obj.get_data()
    acc_info = acc_obj.get_landmarks()
    acc_img = acc_obj.get_data()

    # Calculate the mustache location
    # Used Landmarks:  RIGHT_EYE_POINT: 36, RIGHT_EYE_POINT: 45, RIGHT_BROW_POINT: 19, LEFT_BROW_POINT: 24

    right_head_center = landmarks[36]
    # ((landmarks[17][0] + landmarks[36][0]) / 2, (landmarks[17][1] + landmarks[36][1]) / 2)
    left_head_center = landmarks[45]
    # ((landmarks[26][0] + landmarks[45][0]) / 2, (landmarks[26][1] + landmarks[45][1]) / 2)

    right_head_bottom = landmarks[19]
    left_head_bottom = landmarks[24]

    # Calculate the scale ratio
    scale = distance_points(left_head_center, right_head_center) / distance_points(acc_info[2], acc_info[3])

    center_head_img = (
    (right_head_center[0] + left_head_center[0]) / 2, (right_head_bottom[1] + left_head_bottom[1]) / 2)
    center_head_obj = (acc_info[1][0], acc_info[1][1])

    roll, pose_vect = src_obj.get_face_pose()
    M = cv2.getRotationMatrix2D(center_head_obj, roll, scale)

    new_center = [0.0, 0.0]
    new_center[0] = center_head_obj[0] * M[0][0] + center_head_obj[1] * M[0][1]
    new_center[1] = center_head_obj[0] * M[1][0] + center_head_obj[1] * M[1][1]
    M[0][2] = center_head_img[0] - new_center[0]
    M[1][2] = center_head_img[1] - new_center[1]

    # acc_img = acc_info[0]

    warp = cv2.warpAffine(acc_img, M,
                          (img.shape[1], img.shape[0]))  # Affine transform and then Convert png image to RGB image

    # Get the mask image with alpha channel of object image
    warp_mask = cv2.warpAffine(acc_img[:, :, 3], M, (img.shape[1], img.shape[0]))

    overlay = numpy.zeros(img.shape)
    for c in range(0, 3):
        overlay[:, :, c] = (img[:, :, c] * (1.0 - (warp_mask[:, :] / 255.0))) + warp[:, :, c] * (
                    warp_mask[:, :] / 255.0)

    # Covert float32 numpy array to uint8
    overlay = overlay.astype(numpy.uint8)

    return overlay


def overlay_hat(src_obj, acc_obj):
    landmarks = src_obj.get_landmarks()
    img = src_obj.get_data()
    acc_info = acc_obj.get_landmarks()
    acc_img = acc_obj.get_data()

    # Calculate the mustache location
    # Used Landmarks:  RIGHT_EYE_POINT: 36, RIGHT_EYE_POINT: 45, RIGHT_BROW_POINT: 19, LEFT_BROW_POINT: 24

    right_head_center = landmarks[36]
    # ((landmarks[17][0] + landmarks[36][0]) / 2, (landmarks[17][1] + landmarks[36][1]) / 2)
    left_head_center = landmarks[45]
    # ((landmarks[26][0] + landmarks[45][0]) / 2, (landmarks[26][1] + landmarks[45][1]) / 2)
    right_head_bottom = landmarks[19]
    left_head_bottom = landmarks[24]

    # Calculate the scale ratio
    scale = distance_points(left_head_center, right_head_center) / distance_points(acc_info[2], acc_info[3])

    center_head_img = (
    (right_head_center[0] + left_head_center[0]) / 2, (right_head_bottom[1] + left_head_bottom[1]) / 2)
    center_head_obj = (acc_info[1][0], acc_info[1][1])

    roll, pose_vect = src_obj.get_face_pose()
    M = cv2.getRotationMatrix2D(center_head_obj, roll, scale)

    new_center = [0.0, 0.0]
    new_center[0] = center_head_obj[0] * M[0][0] + center_head_obj[1] * M[0][1]
    new_center[1] = center_head_obj[0] * M[1][0] + center_head_obj[1] * M[1][1]
    M[0][2] = center_head_img[0] - new_center[0]
    M[1][2] = center_head_img[1] - new_center[1]

    # acc_img = acc_info[0]

    warp = cv2.warpAffine(acc_img, M,
                          (img.shape[1], img.shape[0]))  # Affine transform and then Convert png image to RGB image

    # Get the mask image with alpha channel of object image
    warp_mask = cv2.warpAffine(acc_img[:, :, 3], M, (img.shape[1], img.shape[0]))

    overlay = numpy.zeros(img.shape)
    for c in range(0, 3):
        overlay[:, :, c] = (img[:, :, c] * (1.0 - (warp_mask[:, :] / 255.0))) + warp[:, :, c] * (
                    warp_mask[:, :] / 255.0)

    # Covert float32 numpy array to uint8
    overlay = overlay.astype(numpy.uint8)

    return overlay


# Main function for overlay with objects
def overlay_accessory(src_obj, acc_obj):
    if acc_obj.type == 'accessory' and acc_obj.sub_type == 'glasses':
        return overlay_glasses(src_obj, acc_obj)
