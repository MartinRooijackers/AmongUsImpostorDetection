# Computer vision framework for Among us
# Current author(s): Martin Rooijackers
#
#
# Text detection taken from: https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py
# OCR done with Tesseract

# Python code to reading an image using OpenCV
import numpy as np
import cv2

import math
import argparse
import os

import glob

from dataclasses import dataclass

from Levenshtein import distance as lev

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

import utils


@dataclass
class ChatLog:
    id: int
    r: int
    g: int
    b: int
    colorName: str
    frameCount: int
    name: str
    message: str


## global variables

# THe text detector. Currently the EAST trained Neural network
detector = None

# A list of all the chat messages extracted through OCR. See Chatlog for the datastructure
ChatLogArray = []

# TestNumber = 0

##### begin settings that need to be changed   ######

# path to the install location of tesseract. Make sure to include the / at the end as well
Tesseract_location = "D:/Tesseract-OCR/"

# Path to a .pb file contains trained detector network.'
model_detector = "D:/StellarisAI/frozen_east_text_detection.pb"

# folder which contains all the videos you want to analyze. Make sure to include the / at the end as well
video_location = "D:/PythonCVtest/DataSets/Youtube/HarroVideos2/"
# videoLocation = "D:/PythonCVtest/DataSets/Youtube/crew/"


##### end settings that need to be changed   ######

# Colors = [("YELLOW",199,200,69)]

'''
class VideoState(Enum):
    Initialize = 0
    LookingForStart = 1
    LookingForEnd = 2
    Processing = 3
'''

VideoState_LookingForStart = 1
VideoState_LookingForEnd = 2

CheckVotingState_LookingForVote = 1  # look to see if there is a voting going on
CheckVotingState_LookingForConfirm = 2  # a voting has been done, confirm if it is an impostor


############ color functions ############


def grab_colors(img):
    # will show the image in a window

    red = (197, 17, 17)
    lime = (80, 239, 58)
    black = (63, 71, 78)
    purple = (108, 46, 188)
    orange = (239, 124, 12)
    cyan = (57, 255, 221)
    green = (18, 127, 45)
    pink = (240, 84, 189)
    yellow = (244, 245, 84)
    blue = (18, 44, 212)
    white = (214, 222, 241)
    brown = (113, 73, 30)

    color_array = [red] + [lime] + [black] + [purple] + [orange] + [cyan] + [green]
    color_array = color_array + [pink] + [yellow] + [blue] + [white] + [brown]

    for currentColor in color_array:
        height, width, channel = img.shape
        print('width:  ', width)
        print('height: ', height)
        print('channel:', channel)

        total_color = 0
        count_x = 0
        count_y = 0
        for y in range(0, height):
            for x in range(0, width):
                b, g, r = img[y, x]
                # print(b,g,r)
                if currentColor[0] == r and currentColor[1] == g and currentColor[2] == b:
                    print("color found")
                    count_x += x
                    count_y += y
                    total_color += 1
                    # pass
                else:
                    pass
                    # img[y, x] = [255,0,0]

    dim = (800, 600)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('image', resized)
    cv2.imshow('image', img)


def text_detection(frame):
    # Read and store arguments
    conf_threshold = 0.3  # Confidence threshold.
    nms_threshold = 0.2  # Non-maximum suppression threshold.
    inp_width = 1920 - 320  # resize  width (mutliple of 32)
    inp_height = 1056 + 32  # resize  height (mutliple of 32)

    # Create a new named window
    k_win_name = "EAST: An Efficient and Accurate Scene Text Detector"
    cv2.namedWindow(k_win_name, cv2.WINDOW_NORMAL)
    out_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Get frame height and width
    height_ = frame.shape[0]
    width_ = frame.shape[1]
    r_w = width_ / float(inp_width)
    r_h = height_ / float(inp_height)

    # Create a 4D blob from frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (inp_width, inp_height), (123.68, 116.78, 103.94), True, False)

    # Run the detection model
    detector.setInput(blob)

    # tickmeter.start()
    outs = detector.forward(out_names)
    # tickmeter.stop()

    # Get scores and geometry
    scores = outs[0]
    geometry = outs[1]
    [boxes, confidences] = utils.decode_bounding_boxes(scores, geometry, conf_threshold)

    # Apply NMS
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])
        # scale the bounding box coordinates based on the respective ratios
        for j in range(4):
            vertices[j][0] *= r_w
            vertices[j][1] *= r_h

        # get cropped image using perspective transform
        if True:
            cropped = utils.four_points_transform(frame, vertices)
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # Create a 4D blob from cropped image
            blob = cv2.dnn.blobFromImage(cropped, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
            # recognizer.setInput(blob)

            # Run the recognition model
            # tickmeter.start()
            # result = recognizer.forward()
            # tickmeter.stop()

            # decode the result into text
            # wordRecognized = decodeText(result)
            # cv.putText(frame, wordRecognized, (int(vertices[1][0]), int(vertices[1][1])), cv.FONT_HERSHEY_SIMPLEX,
            # 0.5, (255, 0, 0))

        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

    # Put efficiency information
    # label = 'Inference time: %.2f ms' % (tickmeter.getTimeMilli())
    # cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    # Display the frame
    cv2.imshow(k_win_name, frame)


# function that grabs the game settings given the screen that shows them
def grab_settings(frame):
    crop_img = frame[0:1000, 0:500]  # where the setting text is in 1920 x 1080
    crop_img = cv2.bitwise_not(crop_img)  # invert the image for Tesseract
    OCRResults = get_text_from_image(crop_img)
    return OCRResults


# Turn the image into a binary black/white image so Tesseract's OCR works better on it
def preprocess_for_OCR(img):
    # turn the image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshhold
    ret, bin = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # closing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
    # invert black/white
    inv = cv2.bitwise_not(closing)
    return inv


# Function that performs OCR on an image
# works best when the background is white and the text is black (binary image)
def get_text_from_image(img):
    img_output_loc = Tesseract_location + "test.png"
    cv2.imwrite(img_output_loc, img)
    cmd = Tesseract_location + "tesseract.exe " + Tesseract_location + "test.png "
    cmd = cmd + Tesseract_location + "test --psm 6"
    os.system(cmd)  # currently using os call for tesseract.
    ocr_result_loc = Tesseract_location + "test.txt"
    f = open(ocr_result_loc, "r", encoding='utf-8')  # read the results
    OCR_result = f.read()
    return OCR_result


# function that performs OCR on the chat screen
# The chat messages are associated with the color of the person who wrote this
# the information is logged in ChatLogArray , duplicates are not stored
# input: an image containing a chat screen in Among Us. Only works for 1920x1080 at the moment
def extract_text_chat_screen(frame, frame_count=0):
    # cv2.imshow("crop_img", crop_img)

    global ChatLogArray
    img = frame.copy()

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    ret, bin = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    # closing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)

    # invert black/white
    inv = cv2.bitwise_not(closing)
    # cv2.imshow("img_outline", closing)
    # cv2.waitKey()

    height = 1080

    # a boolean which indicated if the messages from the
    # person playing the game should also be included
    include_player_messages = True

    # because the iterator might need to change, don't use for loop
    y = 0  # start at the top
    while y < 750:  # stop when the chat input box is reached
        y = y + 1
        if y > 750:  # reached the end of chat. So stop here
            break
        x = 288  # in the case of 1920 x 1080

        is_message_box = False  # Check if there is a message box at this y
        if closing[y, x] == 255 and closing[y + 100, x] == 255:
            is_message_box = True

        if is_message_box:
            end_y = y + 10
            while closing[end_y, x] == 255:
                end_y = end_y + 1

            crop_img_text = closing[y + 50:end_y - 2, 420:1253]
            if crop_img_text.size == 0:
                print("failure in grabbing text from chat")
                y = end_y + 10
                continue

            OCR_result = get_text_from_image(crop_img_text)

            # how many pixels to move downwards to find the color of the person
            color_person_y = 50
            # get the color values of the player saying the word
            b, g, r = frame[y + color_person_y, x + 100]
            #  get the color name based on the RGB values
            color_name = utils.get_color_name(r, g, b)

            #  variable which indicates if a chat message is already logged
            #  This is done to prevent duplicates
            already_logged = False
            for chat in ChatLogArray:

                if color_name == chat.colorName:
                    if chat.message == OCR_result:
                        already_logged = True
                    # Also check if the levenshtein distance == 1 in case of OCR errors
                    # if the distance is only 1, then it very likely saw the same message,
                    # which is usually caused by an OCR error reading a character the wrong way
                    levenshtein_distance = lev(chat.message, OCR_result)
                    if levenshtein_distance == 1:
                        already_logged = True

            # if the chat entry doesn't exist yet, create it
            if not already_logged:
                # now get the name through OCR
                crop_img = closing[y:y + 50, 420:1253]

                OCR_result_name = get_text_from_image(crop_img)

                # now put it all in the chat logger
                new_log = ChatLog(len(ChatLogArray), r, g, b, color_name, frame_count, OCR_result_name, OCR_result)
                ChatLogArray.append(new_log)
            y = end_y + 10

    # now do the same for the player's chat messages.
    y = 0
    while (y < 750):
        y = y + 1
        if y > 750:  # reached the end of chat. So stop here
            break
        x_player = 1375  # player messages in the case of 1920 x 1080

        is_message_box_player = False
        if closing[y, x_player] == 255 and closing[y + 100, x_player] == 255 and include_player_messages:
            is_message_box_player = True

        if is_message_box_player:
            end_y = y + 10
            while closing[end_y, x_player] == 255:
                end_y = end_y + 1

            crop_img_text = closing[y + 50:end_y - 2, 420:1253]
            if crop_img_text.size == 0:
                print("failure in grabbing text from chat")
                y = end_y + 10
                continue

            OCR_result = get_text_from_image(crop_img_text)

            # how many pixels to move downwards to find the color of the person
            color_person_y = 50

            # get the color values of the player saying the word
            b, g, r = frame[y + color_person_y, x_player - 40]

            color_name = utils.get_color_name(r, g, b)

            already_logged = False
            for chat in ChatLogArray:
                # and chat.r == r and chat.g == g and chat.b == b
                # check if this crewmate already said this (don't log double)

                if color_name == chat.colorName:
                    if chat.message == OCR_result:
                        already_logged = True
                    # Also check if the levenshtein distance == 1 in case of OCR errors
                    # if the distance is only 1, then it very likely saw the same message,
                    # which is usually caused by an OCR error reading a character the wrong way
                    levenshtein_distance = lev(chat.message, OCR_result)
                    if levenshtein_distance == 1:
                        already_logged = True

                # if chat.r == r and chat.g == g and chat.b == b:
                # already_logged = True

            # if the chat entry doesn't exist yet. Create it
            if not already_logged:
                # now get the name through OCR
                # crop_img = closing[ y+50:y+100, 420:1253]
                crop_img = closing[y:y + 50, 420:1253]

                OCR_result_name = get_text_from_image(crop_img)

                # now put it all in the chat logger
                new_log = ChatLog(len(ChatLogArray), r, g, b, color_name, frame_count, OCR_result_name, OCR_result)
                ChatLogArray.append(new_log)
                # print(ChatLogArray)
            # cv2.floodFill()
            # cv2.imshow("img_outline", crop_img)
            y = end_y + 10
            # print("new y: " ,y)
            # cv2.waitKey()

    # cv2.waitKey()


# This function checks if the given frame is a chat screen.
# Currently done by looking at the x/100 in the chat screen
# if so, it does OCR
def check_if_chat_screen(frame, frame_count=0):
    img = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)
    crop_img = img[822:850, 1338:1410]  # location where the x/100 should be in 1920x1080
    OCR_result = get_text_from_image(crop_img)

    # if OCRresult != "":
    # print(OCRresult) #print them
    if "100" in OCR_result:
        extract_text_chat_screen(img, frame_count)


# Function which checks if this screen is the start of the round screen
# For an example of such screen, check the Among Us wiki: https://among-us.fandom.com/wiki/Crewmate
# Should also be able to check if you are starting as a crewmate or impostor
# returns 0 if it is not a start screen, otherwise returns how many impostors there are
def check_if_start_screen(frame):
    img = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)
    total_black_dots = 0
    total_cyan_dots = 0
    for x in range(370, 1550):
        y = 300
        b, g, r = img[y, x]
        if b < 10 and g < 10 and r < 10:
            total_black_dots = total_black_dots + 1
        if (r < 160) and (r > 120) and (g > 230) and (b > 230):
            total_cyan_dots = total_cyan_dots + 1

    # total 1180 dots/pixels
    is_crewmate = False  # check if this screen indicates that the game is starting with the player as crewmate
    if total_black_dots > 100 and total_cyan_dots > 10:
        is_crewmate = True

    if is_crewmate:
        print("found crew frame")

        crop_img = img[400:475, 417:1508]  # currently testing for 1920 x 1080
        crop_img = preprocess_for_OCR(crop_img)
        OCR_result = get_text_from_image(crop_img)

        if "1" in OCR_result:
            return 1
        if "2" in OCR_result:
            return 2
        if "3" in OCR_result:
            return 3
        # in case the OCR has trouble recognizing numbers
        # if there is an "is" , then there is only 1 impostor
        if "is" in OCR_result:
            return 1

        # two common OCR errors, should be 2
        if " e " in OCR_result:
            return 2
        if " ec " in OCR_result:
            return 2

        # are could mean 2 or 3
        if "are" in OCR_result:
            # might also be 3, but OCR errors seems to mostly occur with 2
            # Can be skipped if need be
            return 2

    return 0


# Function which checks if this screen is the end of the round screen
# returns ""Neither"" if it is not an end screen,
# otherwise returns "defeat" or "victory" to indicate what kind of screen it is
def check_if_end_screen(frame):
    # check y = 225
    # the higher up the y (check), the more likely it is that the end screen is still fading in

    img = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)
    output = "neither"

    total_black_dots = 0
    total_red_dots = 0
    total_blue_dots = 0
    for x in range(230, 1676):
        y = 225
        b, g, r = img[y, x]
        if b < 10 and g < 10 and r < 10:
            total_black_dots = total_black_dots + 1
        if r < 10 and (115 < g < 150) and b > 215:
            total_blue_dots = total_blue_dots + 1
        if (r > 230) and (g < 10) and (b < 10):
            total_red_dots = total_red_dots + 1

    if total_black_dots > 200 and total_red_dots > 15:
        output = "defeat"

    if total_black_dots > 200 and total_blue_dots > 10:
        output = "victory"

    return output


#  function which tries to extract the colors of the impostors in a defeat screen
#  Some cosmetics will interfere with how this process currently works
#  Should eventually be changed into something that can also handle cosmetics
#  input: an image containing a defeat screen, and the total number of impostors
#  output: a list of the colors of teh impostors that appear in the image
def grab_colors_defeat_screen(frame, total_impostors):
    # Make sure to check if they are ghosts
    # if they are, then there should be a black pixel at::
    # ghost 1,   x = 874 , y = 734
    # ghost 2  ,  x = 1121 , y = 772=0

    img = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)
    colors = []
    impostor1_ghost = False
    impostor2_ghost = False
    # Haven't seen loc of ghost 3 yet, once I do I will add it

    # check if impostor 1 is a ghost
    b, g, r = img[734, 874]
    if (b < 30) and (g < 30) and (r < 30):
        impostor1_ghost = True

    # check if impostor 2 is a ghost
    b, g, r = img[770, 1121]
    if (b < 30) and (g < 30) and (r < 30):
        impostor2_ghost = True

    # 1st (and only) impostor: x = 1000 , y = 725
    if total_impostors == 1:
        b, g, r = img[725, 1000]
        color_name = utils.get_color_name(r, g, b)
        if impostor1_ghost:
            color_name = utils.get_ghost_color_name(r, g, b)
        colors = colors + [color_name]

    # 1st impostor , x = 1000 , y = 725
    # 2nd impostor, x = 1110 , y = 700
    if total_impostors == 2:
        # 1st impostor
        b, g, r = img[725, 1000]
        color_name = utils.get_color_name(r, g, b)
        if impostor1_ghost:
            color_name = utils.get_ghost_color_name(r, g, b)
        colors = colors + [color_name]
        # 2nd impostor
        b, g, r = img[700, 1110]
        color_name = utils.get_color_name(r, g, b)
        if impostor2_ghost:
            color_name = utils.get_ghost_color_name(r, g, b)
        colors = colors + [color_name]

    # 1st impostor =   x= 1010, y = 723
    # 2nd impostor ,  x=  1157, y = 700
    # 3rd impostor ,   x = 832 , y = 692
    if total_impostors == 3:
        # 1st impostor
        b, g, r = img[723, 1010]
        color_name = utils.get_color_name(r, g, b)
        if impostor1_ghost:
            color_name = utils.get_ghost_color_name(r, g, b)
        colors = colors + [color_name]
        # 2nd impostor
        b, g, r = img[700, 1157]
        color_name = utils.get_color_name(r, g, b)
        if impostor2_ghost:
            color_name = utils.get_ghost_color_name(r, g, b)
        colors = colors + [color_name]
        # 3th impostor
        b, g, r = img[692, 832]
        color_name = utils.get_color_name(r, g, b)
        colors = colors + [color_name]

    return colors


#  functino which generates the .txt file with the impostor info
#  one integer with the number of impostors followed by the list of color from the input seperated by newlines
#  input: the location of where to put the .txt, the list of impostor colors, the number of impostors
def output_impostor_data(output_loc, Impostor_colors, number_of_impostors):
    # in case the function gets the .json location with this file is related to
    output_loc = output_loc.replace(".json", ".txt")

    f = open(output_loc, "w")
    number_write = str(number_of_impostors)
    f.write(number_write)
    f.write("\n")
    for color in Impostor_colors:
        f.write(color)
        f.write("\n")


#  function which outputs the ChatLogArray into a .json file
#  input: the location of where to put the .json file
def output_OCR_data(outputlocation):
    global ChatLogArray

    f = open(outputlocation, "w")
    total_iterations = 0
    loop_frame = 0
    if len(ChatLogArray) > 0:
        loop_frame = ChatLogArray[0].frameCount

    meeting_id = 1
    f.write("{\n \"ChatMessages\": [")
    for chat in ChatLogArray:
        total_iterations = total_iterations + 1
        f.write("  {\n")
        f.write("    \"ID\": ")
        f.write(str(chat.id))
        f.write(",\n")

        f.write("    \"RED\": ")
        f.write(str(chat.r))
        f.write(",\n")

        f.write("    \"GREEN\": ")
        f.write(str(chat.g))
        f.write(",\n")

        f.write("    \"BLUE\": ")
        f.write(str(chat.b))
        f.write(",\n")

        f.write("    \"FRAMECOUNT\": ")
        f.write(str(chat.frameCount))
        f.write(",\n")

        # current method of determining meetingID
        if chat.frameCount - loop_frame > 2000:
            meeting_id = meeting_id + 1

        loop_frame = chat.frameCount

        f.write("    \"meeting_id\": ")
        f.write(str(meeting_id))
        f.write(",\n")

        f.write("    \"COLORNAME\": \"")
        f.write(str(chat.colorName))
        f.write("\",\n")

        name_write = chat.name.replace("\n\x0c", "").replace("\n", "").replace("\"", "\\\"")
        name_write = "    \"NAME\": \"" + name_write + "\",\n"
        f.write(name_write)
        textWrite = chat.message.replace("\n\x0c", "").replace("\n", "").replace("\"", "\\\"")
        textWrite = "    \"MESSAGE\": \"" + textWrite + "\"\n"
        f.write(textWrite)
        if total_iterations == len(ChatLogArray):
            f.write("  }\n")
        else:
            f.write("  },\n")

    f.write("  ]\n}\n")

    f.close()


# determine who is getting voted out, or if the vote will be a tie/skipped
# input: an image with the voting screen with the votes being displayed
def tally_votes(frame):
    frame = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)

    result = "Nothing"

    step_vote_icon = 459 - 404  # step size in the x direction between each vote
    max_votes = 0
    current_voted = "Nothing"

    first_x = 322  # location of the player colors in the left side of the vote screen
    second_x = 980  # location of the player colors in the right side of the vote screen
    # x,y for voting screen,  x+y for color , then x,y for start of voting icons
    all_vote_positions = [(284, 222, first_x, 282, 415, 291), (277, 359, first_x, 414, 415, 428),
                          (282, 494, first_x, 562, 415, 565), (282, 632, first_x, 687, 415, 705)]
    all_vote_positions = all_vote_positions + [(275, 766, first_x, 834, 415, 841), (935, 219, second_x, 282, 1070, 293),
                                               (935, 359, 1006, 436, 1070, 428)]
    all_vote_positions = all_vote_positions + [(936, 495, second_x, 543, 1070, 565),
                                               (933, 620, second_x, 693, 1070, 705),
                                               (941, 770, second_x, 830, 1070, 841)]

    # original 2nd right side (935,359,second_x,411,1070,428)
    # changed because the broken screen as ghost can interfere

    for pos in all_vote_positions:

        x = pos[2]
        y = pos[3]
        b, g, r = frame[y, x]
        vote_color = utils.get_color_name(r, g, b)
        x = pos[0]
        y = pos[1]
        b, g, r = frame[y, x]
        # check if the player is actually in the game
        if (b > 200) and (g > 200) and (r > 200):
            print("Still in Game")
            # now count the total amount of people who voted for this player
            TotalVotes = 0
            for i in range(7):
                x = pos[4] + (step_vote_icon * i)
                y = pos[5]
                b, g, r = frame[y][x]
                if (b > 220) and (r > 200) and (g > 200):  # no more votes for this person
                    break
                TotalVotes = TotalVotes + 1
                print(TotalVotes)
                if TotalVotes == max_votes:  # Tie
                    current_voted = "Nothing"
                if TotalVotes > max_votes:  # current most voted
                    max_votes = TotalVotes
                    current_voted = vote_color

    result = current_voted

    # also check the amount of votes for skipping
    # vote icons for that start at x = 430, y = 960  inn 1920x1080
    skip_step_size = 52  # x distance between each vote icon for skipping vote
    x = 430
    y = 960

    total_vote_skip = 0
    for i in range(7):
        x = 430 + (i * skip_step_size)
        b, g, r = frame[y][x]
        if r > 150:
            break
        total_vote_skip = total_vote_skip + 1
    if total_vote_skip >= max_votes:
        result = "Nothing"

    return result


# function which checks if a voting has occurred
def find_votes(frame):
    img = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)
    # check if this is a voting screen
    total_grey_dots = 0
    x = 1602
    for y in range(67, 1010):
        b, g, r = img[y, x]
        if (b > 140) and (b < 170) and (r > 130) and (r < 170) and (g < 170) and (g > 130):
            total_grey_dots = total_grey_dots + 1

    if total_grey_dots < 500:
        return "Nothing"

    crop_img = img[910:967, 1230:1555]  # currently testing for 1920 x 1080
    height, width, channel = crop_img.shape

    for x in range(0, width):
        for y in range(0, height):
            b, g, r = crop_img[y, x]
            if (b > 200) and (g > 200) and (r > 200):  # white
                crop_img[y, x] = (0, 0, 0)
            elif (b < 50) and (g < 50) and (r > 190):  # red
                crop_img[y, x] = (0, 0, 0)
            else:
                crop_img[y, x] = (255, 255, 255)

    inv = cv2.bitwise_not(crop_img)
    gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    #     ret, bin = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    # threshold
    ret, bin = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    # closing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
    # invert black/white
    inv = cv2.bitwise_not(closing)

    OCR_result = get_text_from_image(inv)

    # actually "Proceeding"  , but the screen has a broken look if you are a ghost
    if "eeding" in OCR_result:
        # votes will all appear at 2seconds , so any time after will allow a tally of them
        if ("2" in OCR_result) or ("1" in OCR_result) or ("0" in OCR_result):
            # count votes
            results_tally = tally_votes(img)
            print(results_tally)
            # cv2.imshow("vote screen original", frame)
            # cv2.waitKey(0)
            return results_tally

    return "Nothing"


# A function which checks if the person being ejected is an impostor
# this function will only work correctly if the "confirm ejects" option is turned on in the game being analyzed
# returns true if the player being ejected is not an impostor (given the current frame).
# False otherwise
def check_impostor_eject(frame):
    img = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)
    crop_img = img[535:603, 490:1456]  # currently testing for 1920 x 1080

    height, width, channel = crop_img.shape

    # invert the colors and turn the image into a binary image for better OCR
    for x in range(0, width):
        for y in range(0, height):
            b, g, r = crop_img[y, x]
            if (b > 200) and (g > 200) and (r > 200):  # white
                crop_img[y, x] = (0, 0, 0)
            else:
                crop_img[y, x] = (255, 255, 255)

    OCR_result = get_text_from_image(crop_img)

    # print(OCR_result)
    # cv2.imshow("eject text original", frame)
    # cv2.imshow("eject text", crop_img)
    # cv2.waitKey(0)

    # for when there are multiple impostors
    if "not An Impostor" in OCR_result:
        return True

    # for when there is 1 impostor
    if "not The Impostor" in OCR_result:
        return True

    return False


# check if the frame/image represents a vote screen
# returns true if that is the case
# returns false otherwise
def check_if_vote_screen(frame):
    # grey color in voting screen:  r = 146, g = 156 , b = 170

    img = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)

    # check if this is a voting screen
    total_grey_dots = 0
    x = 1602
    for y in range(60, 400):
        b, g, r = img[y, x]
        if (b > 150) and (b < 180) and (r > 130) and (r < 170) and (g < 170) and (g > 130):
            total_grey_dots = total_grey_dots + 1

    if total_grey_dots < 300:
        return False

    return True


# checks if a voting screen is fading into the result of the vote
# used to more easily detect a voting screen where people have voted
def check_if_vote_screen_fading(frame):
    # check the pixel at   x = 1602 , y = 100

    img = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)

    x = 1602
    y = 100

    b, g, r = img[y, x]

    # regular grey:  r = 146, g = 156 , b = 170
    if (b < 120) and (r < 120) and (g < 120):
        return True

    return False


#  the main function for processing a single video
#  this function calls the other functions to analyze the video to extract the information needed for classification
#  input: location = where the video is located , outputlocation = where to output the informatio gathered
def process_video(location, output_location):
    # print(location)
    # clear previous data
    global ChatLogArray
    ChatLogArray = []

    cap = cv2.VideoCapture(location)

    # make a special folder for this video for putting all the data in
    if not os.path.isdir(output_location):
        os.mkdir(output_location)

    total_games_processed = 1

    current_state = VideoState_LookingForStart
    total_impostors = 0

    current_voting_check_state = CheckVotingState_LookingForVote
    current_color_vote_check = "Nothing"  # the color to check if it actually is an impostor (when confirm eject is on)
    frame_star_checking = 0  # the frame where the algorithm starts checking for eject confirmation
    impostor_list_voting = []

    check_grey_to_black = False

    current_frame = 0

    while cap.isOpened():
        # print("next frame")
        # Read frame
        has_frame, frame = cap.read()
        current_frame = current_frame + 1
        if not has_frame:
            print("Done processing this video")
            break

        if current_voting_check_state == CheckVotingState_LookingForConfirm:

            max_frames_to_check = frame_star_checking + (30 * 20)  # check for a max of 600 frames
            if current_frame > max_frames_to_check:
                current_voting_check_state = CheckVotingState_LookingForVote
                # after all this time, still no message found that the player was not an impostor
                # so add the player to the impostor list
                impostor_list_voting = impostor_list_voting + [current_color_vote_check]
                current_color_vote_check = "Nothing"

            player_not_impostor = check_impostor_eject(frame)
            if player_not_impostor:
                current_color_vote_check = "Nothing"
                current_voting_check_state = CheckVotingState_LookingForVote

        if current_state == VideoState_LookingForStart:
            total_impostors = check_if_start_screen(frame)

        if total_impostors > 0 and current_state == VideoState_LookingForStart:
            current_state = VideoState_LookingForEnd
            start_screen_debug_loc = output_location + "/" + str(total_games_processed) + ".Start.png"
            cv2.imwrite(start_screen_debug_loc, frame)

        # only check for votes when a new game is being analyed
        # otherwise you will be spending time checking for votes in Impostor games as well, which we skip
        if (current_voting_check_state == CheckVotingState_LookingForVote) and (
                current_state == VideoState_LookingForEnd) and (not check_grey_to_black):
            check_grey_to_black = check_if_vote_screen(frame)

        # begin rollback in video to check votes
        if check_grey_to_black:
            # resized_image =
            is_fading = check_if_vote_screen_fading(frame)
            if is_fading:
                # if it is fading, that must mean that the count has happened
                # so roll back to there to start the count
                # print("found fading frame")
                # cv2.imshow("fading vote", frame)
                # cv2.waitKey(0)
                get_current_frame = current_frame  # useful for rolling back
                rollback_frame = current_frame
                resized_image = cv2.resize(frame, (1920, 1080))
                b, g, r = resized_image[100, 1602]
                # now keep going back till there is a good overview of the votes
                # cap.set(cv2.CAP_PROP_POS_FRAMES, 2200)
                best_score = b + g + r
                best_frame = current_frame

                getting_brighter = True
                # while getting_brighter:
                for countF in range(1, 30):
                    # getting_brighter = False
                    # print("Getting more brighter")
                    rollback_frame = rollback_frame - countF
                    cap.set(cv2.CAP_PROP_POS_FRAMES, rollback_frame)
                    rollback_has_frame, frame_roll_back = cap.read()
                    if not rollback_has_frame:
                        break
                    new_resized_image = cv2.resize(frame_roll_back, (1920, 1080))
                    new_b, new_g, new_r = new_resized_image[100, 1602]

                    # check if the chat screen is still visible
                    chat_b, chat_g, chat_r = new_resized_image[105, 1549]
                    # if so, skip this frame if the picture is bright enough

                    # if any of the color channels get lower, or all are the same
                    # then you can stop searching

                    get_score = int(new_b) + int(new_g) + int(new_r)

                    expected_value = 140 + 151 + 164
                    # print((expected_value - get_score))
                    if (chat_b > 220) and (chat_g > 220) and (chat_r > 220) and (expected_value - get_score) < 20:
                        continue

                    if get_score > best_score:
                        # if get_score > best_score and (( get_score - best_score ) > 10):
                        best_score = get_score
                        best_frame = rollback_frame

                cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
                rollback_has_frame, frame_roll_back = cap.read()
                # print("tally voted with rollback frame")
                # cv2.imshow("tally vote", frame_roll_back)
                # cv2.waitKey(0)

                result_tally = tally_votes(frame_roll_back)
                cap.set(cv2.CAP_PROP_POS_FRAMES, get_current_frame)

                if not ("Nothing" in result_tally):  # a player is being voted out
                    current_voting_check_state = CheckVotingState_LookingForConfirm
                    current_color_vote_check = result_tally
                    frame_star_checking = current_frame
                    # debugInfo
                    vote_debug_loc = output_location + "/" + str(
                        total_games_processed) + current_color_vote_check + ".Vote.png"
                    resized_frame = cv2.resize(frame_roll_back, (1920, 1080))
                    cv2.imwrite(vote_debug_loc, resized_frame)
                else:  # otherwise, skip forward
                    cap.set(cv2.CAP_PROP_POS_FRAMES, get_current_frame + (30 * 5))
                    current_frame = get_current_frame + (30 * 5)
                check_grey_to_black = False

        # end rollback check votes

        # as long as a game is ongoing, do OCR
        if current_state == VideoState_LookingForEnd:
            check_if_chat_screen(frame, current_frame)

        if current_state == VideoState_LookingForEnd:
            result = check_if_end_screen(frame)
            # once the game has ended, output the OCR and start looking for the next game
            if not ("neither" in result):

                # if the game ended, and a color is still under consideration for impostor status
                # then that color must be the impostor
                if not ("Nothing" in current_color_vote_check):
                    impostor_list_voting = impostor_list_voting + [current_color_vote_check]

                end_screen_debug_loc = output_location + "/" + str(total_games_processed) + "End.png"
                resized_frame = cv2.resize(frame, (1920, 1080))
                cv2.imwrite(end_screen_debug_loc, resized_frame)
                # cv2.imwrite(end_screen_debug_loc, frame)

                # total_games_processed = total_games_processed + 1
                output_OCR_loc = output_location + "/" + str(total_games_processed) + ".json"
                current_state = VideoState_LookingForStart
                output_OCR_data(output_OCR_loc)
                ChatLogArray = []
                # if there is a defeat screen, grab the impostor data from there
                output_impostor_loc = output_location + "/" + str(total_games_processed) + ".txt"
                if "defeat" in result:
                    colors_impostors = grab_colors_defeat_screen(frame, total_impostors)
                    output_impostor_data(output_impostor_loc, colors_impostors, total_impostors)
                    # outputImpostorData(output_impostor_loc, impostor_list_voting, total_impostors)

                    print(colors_impostors)
                    # cv2.imshow("crew screen", frame)
                    # cv2.waitKey(0)
                else:  # otherwise, rely on voting data

                    output_impostor_data(output_impostor_loc, impostor_list_voting, total_impostors)
                total_games_processed = total_games_processed + 1
                # reset some more variables
                current_voting_check_state = CheckVotingState_LookingForVote
                current_color_vote_check = "Nothing"  # the color to check if it actually is an impostor
                impostor_list_voting = []
                check_grey_to_black = False

        # skip 15 frames
        # TO maybe DO: use grab() instead so OpenCV doesn't have to decode
        frames_to_skip = 15

        while frames_to_skip > 0:
            frames_to_skip = frames_to_skip - 1
            cap.read()
            current_frame = current_frame + 1

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

    return


#  The main function. Here the text detection NN is loaded, and the list of videos is extracted from the location
#  The actual video analysis happens in the ProcessVideo function
def main():
    # Read and store arguments
    confThreshold = 0.3  # Confidence threshold.
    nmsThreshold = 0.2  # Non-maximum suppression threshold.
    inpWidth = 1920 - 320  # resize  width (mutliple of 32)
    inpHeight = 1056 + 32  # resize  height (mutliple of 32)

    # Load network
    global detector
    detector = cv2.dnn.readNet(model_detector)

    videos = os.listdir(video_location)
    for video in videos:
        video_loc = video_location + video
        remove_extension = video.split(".")[0]
        # outputLoc =  videoLocation + video.replace(".mp4",".json")
        outputLoc = video_location + remove_extension + ".json"
        # ProcessVideo(video_loc, outputLoc)
        process_video(video_loc, video_location + remove_extension)
        print(video)
    exit(0)


if __name__ == "__main__":
    main()
