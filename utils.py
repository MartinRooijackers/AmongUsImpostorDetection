from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

import numpy as np
import cv2
import math


def get_hue(r, g, b):
    minimum = min(r, g, b)
    maximum = max(r, g, b)

    if min == max:
        return 0

    hue = 0.0
    if max == r:
        hue = (g - b) / (maximum - minimum)
    if max == g:
        hue = 2.0 + ((b - r) / (maximum - minimum))
    if max == b:
        hue = 4.0 + ((r - g) / (maximum - minimum))

    hue = hue * 60
    if hue < 0.0:
        hue = hue + 360

    return round(hue)


# get color name based on RGB distance
# RGB values of colors are acquired from: https://among-us.fandom.com/wiki/Category:Colors
def get_color_name_RGB(r, g, b):
    red = ("RED", 197, 17, 17)
    lime = ("LIME", 80, 239, 58)
    black = ("BLACK", 63, 71, 78)
    purple = ("PURPLE", 108, 46, 188)
    orange = ("ORANGE", 239, 124, 12)
    cyan = ("CYAN", 57, 255, 221)
    green = ("GREEN", 18, 127, 45)
    pink = ("PINK", 240, 84, 189)
    yellow = ("YELLOW", 244, 245, 84)
    blue = ("BLUE", 18, 44, 212)
    white = ("WHITE", 214, 222, 241)
    brown = ("BROWN", 113, 73, 30)

    color_list = [red] + [lime] + [black] + [purple] + [orange] + [cyan] + [green] + [pink] + [yellow] + [blue] + \
                 [white] + [brown]

    # print(color_list)

    best_match_color = "NONE"
    closest_dist = 99999
    for color in color_list:
        RGB_distance = abs(r - color[1]) + abs(g - color[2]) + abs(b - color[3])
        if RGB_distance < closest_dist:
            best_match_color = color[0]
            closest_dist = RGB_distance

    return best_match_color


# get color name based on deltaE of the cie2000 color space
# RGB values of colors are acquired from: https://among-us.fandom.com/wiki/Category:Colors
def get_color_name(r, g, b):
    red = ("RED", 197, 17, 17)
    lime = ("LIME", 80, 239, 58)
    black = ("BLACK", 63, 71, 78)
    purple = ("PURPLE", 108, 46, 188)
    orange = ("ORANGE", 239, 124, 12)
    cyan = ("CYAN", 57, 255, 221)
    green = ("GREEN", 18, 127, 45)
    pink = ("PINK", 240, 84, 189)
    yellow = ("YELLOW", 244, 245, 84)
    blue = ("BLUE", 18, 44, 212)
    white = ("WHITE", 214, 222, 241)
    brown = ("BROWN", 113, 73, 30)

    color_list = [red] + [lime] + [black] + [purple] + [orange] + [cyan] + [green] + [pink] + [yellow] + [blue] + [
        white] + [brown]

    # print(color_list)

    best_match_color = "NONE"
    closest_dist = 99999

    for color in color_list:
        color1_rgb = sRGBColor(r, g, b, True)
        color2_rgb = sRGBColor(color[1], color[2], color[3], True)
        color1_lab = convert_color(color1_rgb, LabColor)
        color2_lab = convert_color(color2_rgb, LabColor)
        delta_e = delta_e_cie2000(color1_lab, color2_lab)

        if delta_e < closest_dist:
            best_match_color = color[0]
            closest_dist = delta_e

    return best_match_color


#  ghosts have a different RGB color encoding
#  currently still using RGB distance since the defeat screen adds a red hue
#   which could interfere with the deltaE of cie2000
#  blue and purple ghosts look quite alike though. Even manually distinguishing between them is difficult
#  still need to look for a fix for that
def get_ghost_color_name(r, g, b):
    # extracted ghost colors

    red = ("RED", 127, 15, 2)
    lime = ("LIME", 74, 76, 15)
    black = ("BLACK", 80, 28, 28)
    purple = ("PURPLE", 66, 17, 87)
    orange = ("ORANGE", 146, 49, 6)
    cyan = ("CYAN", 80, 104, 92)
    green = ("GREEN", 80, 80, 20)
    pink = ("PINK", 171, 49, 77)
    yellow = ("YELLOW", 101, 61, 20)
    blue = ("BLUE", 64, 21, 97)
    white = ("WHITE", 155, 104, 111)
    brown = ("BROWN", 81, 36, 37)

    # end of extracted ghost colors

    colorList = [red] + [lime] + [black] + [purple] + [orange] + [cyan] + [green] + [pink] + [yellow] + [blue] + [
        white] + [brown]

    # print(colorList)

    best_match_color = "NONE"
    closest_dist = 99999

    for color in colorList:
        # print (color)
        # RNGdistance = 0
        RGB_distance = abs(r - color[1]) + abs(g - color[2]) + abs(b - color[3])
        if RGB_distance < closest_dist:
            best_match_color = color[0]
            closest_dist = RGB_distance

    return best_match_color


############ Utility functions for text detection ############

def four_points_transform(frame, vertices):
    vertices = np.asarray(vertices)
    output_size = (100, 32)
    target_vertices = np.array([
        [0, output_size[1] - 1],
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1]], dtype="float32")

    rotation_matrix = cv2.getPerspectiveTransform(vertices, target_vertices)
    result = cv2.warpPerspective(frame, rotation_matrix, output_size)
    return result


def decode_text(scores):
    text = ""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += '-'

    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return ''.join(char_list)


def decode_bounding_boxes(scores, geometry, score_thresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if score < score_thresh:
                continue

            # Calculate offset
            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = (
                [offset_x + cos_a * x1_data[x] + sin_a * x2_data[x], offset_y - sin_a * x1_data[x] + cos_a * x2_data[x]])

            # Find points for rectangle
            p1 = (-sin_a * h + offset[0], -cos_a * h + offset[1])
            p3 = (-cos_a * w + offset[0], sin_a * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

## end of utility functions for text detection ######
