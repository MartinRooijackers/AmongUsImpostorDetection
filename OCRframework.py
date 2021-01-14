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
## TODO: maybe change this

detector = None

ChatLogArray = []


Colors = [("YELLOW",199,200,69)]



############ Utility functions for colors ############




############ Utility functions for text detection ############

def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result


def decodeText(scores):
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


def decodeBoundingBoxes(scores, geometry, scoreThresh):
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
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


## end of utility functions

def grabcolors():
  img = cv2.imread('D:/PythonCVtest/DetectCrew2.jpg')
  # will show the image in a window

  colorYellow = (199,200,69)
  #colorYellow = (234,234,83)
  colorGreen = (79,237,57)
  colorPurple = (87,38,151)
  colorDarkBlue = (18,34,128)
  #colorBlack =  (46,53,56)
  colorBlack =  (57,66,73)

  colorArray = [colorYellow]
  colorArray = colorArray + [colorGreen]
  colorArray = colorArray + [colorPurple]
  colorArray = colorArray + [colorDarkBlue]
  colorArray = colorArray + [colorBlack]

  print(colorArray)
  for currentColor in colorArray:

      print(currentColor)
      height, width, channel = img.shape
      print('width:  ', width)
      print('height: ', height)
      print('channel:', channel)

      totalColor = 0
      countX = 0
      countY = 0
      for y in range(0,height):
          for x in range(0,width):
              b,g,r = img[y, x]
              #print(b,g,r)
              if currentColor[0] == r and currentColor[1] == g and currentColor[2] == b:
                  print("color found")
                  countX += x
                  countY += y
                  totalColor += 1
                  #pass
              else:
                  pass
                  #img[y, x] = [255,0,0]
      if totalColor > 0:
        averageX = int(countX / totalColor)
        averageY = int(countY / totalColor)
        if currentColor == colorBlack:
            cv2.rectangle(img, (averageX - 125, averageY - 100), (averageX + 75, averageY + 100),(currentColor[2], currentColor[1], currentColor[0]), 10)
            cv2.putText(img,"21%",(averageX - 125,averageY + 140),cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255), 4)
            continue
        cv2.rectangle(img, (averageX - 100, averageY - 100), (averageX + 100, averageY + 100), (currentColor[2], currentColor[1], currentColor[0]), 10)
        if currentColor == colorGreen:
            cv2.putText(img,"me",(averageX - 100,averageY + 140),cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255), 4)
        if currentColor == colorYellow:
            cv2.putText(img,"76%",(averageX - 100,averageY + 140),cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255), 4)


  dim = (800, 600)
  # resize image
  resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
  cv2.imshow('image', resized)
  k = cv2.waitKey(0) & 0xFF

  cv2.imshow('image', img)
  k = cv2.waitKey(0) & 0xFF
  exit(0)


def TextDetection(frame):
  # Read and store arguments
  confThreshold = 0.3  # Confidence threshold.
  nmsThreshold = 0.2  # Non-maximum suppression threshold.
  inpWidth = 1920 - 320  # resize  width (mutliple of 32)
  inpHeight = 1056 + 32  # resize  height (mutliple of 32)

  # Create a new named window
  kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
  cv2.namedWindow(kWinName, cv2.WINDOW_NORMAL)
  outNames = []
  outNames.append("feature_fusion/Conv_7/Sigmoid")
  outNames.append("feature_fusion/concat_3")


  # Get frame height and width
  height_ = frame.shape[0]
  width_ = frame.shape[1]
  rW = width_ / float(inpWidth)
  rH = height_ / float(inpHeight)

  # Create a 4D blob from frame.
  blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

  # Run the detection model
  detector.setInput(blob)

  #tickmeter.start()
  outs = detector.forward(outNames)
  #tickmeter.stop()

  # Get scores and geometry
  scores = outs[0]
  geometry = outs[1]
  [boxes, confidences] = decodeBoundingBoxes(scores, geometry, confThreshold)

  # Apply NMS
  indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
  for i in indices:
    # get 4 corners of the rotated rect
    vertices = cv2.boxPoints(boxes[i[0]])
    # scale the bounding box coordinates based on the respective ratios
    for j in range(4):
        vertices[j][0] *= rW
        vertices[j][1] *= rH

    # get cropped image using perspective transform
    if True:
        cropped = fourPointsTransform(frame, vertices)
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
  cv2.imshow(kWinName, frame)


#function that grabs the game settings given the screen that shows them
# TODO: use text detection to determine the part of the image to grab
def GrabSettings(frame):
    crop_img = frame[0:1000, 0:500] #currently testing for 1920 x 1080
    crop_img = cv2.bitwise_not(crop_img) #invert the image for Tesseract

    cv2.imshow("crop_img", crop_img)
    cv2.imwrite("D:/Tesseract-OCR/test.png", crop_img)
    cmd = "D:/Tesseract-OCR/tesseract.exe  D:/Tesseract-OCR/test.png D:/Tesseract-OCR/test.txt"
    os.system(cmd) #currently using os call for tesseract. TODO: use python tesseract library instead
    f = open("D:/Tesseract-OCR/test.txt", "r") #read the results
    print(f.read()) #print them



def GetColorName(r,g,b):
    red = ("RED",197,17,17)
    lime = ("LIME",80,239,58)
    black = ("BLACK",63,71,78)
    purple = ("PURPLE",108,46,188)
    orange = ("ORANGE",239,124,12)
    cyan = ("CYAN",57,255,221)
    green = ("GREEN",18,127,45)
    pink = ("PINK",240,84,189)
    yellow = ("YELLOW",244,245,84)
    blue = ("BLUE",18,44,212)
    white = ("WHITE",214,222,241)
    brown = ("BROWN",113,73,30)

    colorList = [red] + [lime] + [black] + [purple] + [orange] + [cyan] + [green] + [pink] + [yellow] + [blue] + [white] + [brown]

    #print(colorList)

    bestMatchColor = "NONE"
    closestDist = 99999

    for color in colorList:
        #print (color)
        RNGdistance = 0
        RNGdistance = abs(r - color[1]) + abs(g - color[2]) + abs(b - color[3])
        if RNGdistance < closestDist:
            bestMatchColor = color[0]
            closestDist = RNGdistance

    return bestMatchColor

def ExtractTextChatScreen(frame,frameCount = 0):
    #cv2.imshow("crop_img", crop_img)

    global ChatLogArray


    # convert to grayscale
    img = frame.copy()
    #img = cv2.resize(img, (1920, 1080)) #resize the image to (1920, 1080)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshhold
    ret, bin = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    # closing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)

    # invert black/white
    #inv = cv2.bitwise_not(closing)
    #cv2.imshow("img_outline", closing)

    height = 1080

    #because the iterator might need to change, dont use for loop
    #for y in range(0, height):
    y = 0
    while(y < 750):
        y = y + 1
        if y > 750: #reached the end of chat. So stop here
          break
        x = 288 #in the case of 1920 x 1080
        #for x in range(0, width):
        #b, g, r = closing[y, x]
        #print(closing[y, x])

        #print(y)
        if closing[y, x] == 255 and closing[y+100, x] == 255:
          endY = y + 10
          while closing[endY, x] == 255:
            endY = endY + 1


          #cv2.imshow("img_outline", inv)
          #cv2.waitKey()

          crop_img_text = closing[y + 50 :endY - 2, 420:1253]
          if not crop_img_text.size == 0:
            cv2.imwrite("D:/Tesseract-OCR/test.png", crop_img_text)
            #cv2.imshow("img_outline", crop_img_text)
          else:
              print("failure in grabbing text from chat")
              y = endY + 10
              continue

          # use psm 6 for better ocr
          cmd = "D:/Tesseract-OCR/tesseract.exe  D:/Tesseract-OCR/test.png D:/Tesseract-OCR/test --psm 6"
          os.system(cmd)  # currently using os call for tesseract. TODO: use python tesseract library instead
          f = open("D:/Tesseract-OCR/test.txt", "r", encoding='utf-8')  # read the results
          OCRresult = f.read()
          print(OCRresult)

          # get the color values of the crewmate saying the word
          b, g, r = frame[y + 50, x + 100]

          colorName = GetColorName(r,g,b)

          alreadyLogged = False
          for chat in ChatLogArray:
              # and chat.r == r and chat.g == g and chat.b == b
              #check if this crewmate already said this (don't log double)
              if chat.message == OCRresult and colorName == chat.colorName:
                  alreadyLogged = True
              #if chat.r == r and chat.g == g and chat.b == b:
                  #alreadyLogged = True


          # if the chat entry doesn't exist yet. Create it
          if alreadyLogged == False:

            #now get the name through OCR
            # crop_img = closing[ y+50:y+100, 420:1253]
            crop_img = closing[y:y + 50, 420:1253]
            inv = cv2.bitwise_not(crop_img)

            h, w = inv.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(inv, mask, (0, 0), 255)
            cv2.imwrite("D:/Tesseract-OCR/test.png", inv)
            cmd = "D:/Tesseract-OCR/tesseract.exe  D:/Tesseract-OCR/test.png D:/Tesseract-OCR/test"
            os.system(cmd)  # currently using os call for tesseract. TODO: use python tesseract library instead
            f = open("D:/Tesseract-OCR/test.txt", "r")  # read the results
            OCRresultName = f.read()

            #now put it all in the chat logger
            newLog = ChatLog(len(ChatLogArray),r,g,b,colorName,frameCount,OCRresultName,OCRresult)
            ChatLogArray = ChatLogArray + [newLog]
            print(ChatLogArray)
          #cv2.floodFill()
          #cv2.imshow("img_outline", crop_img)
          y = endY + 10
          #print("new y: " ,y)
          #cv2.waitKey()

    #cv2.waitKey()


#This function checks if the given frame is a chat screen.
# Currently done by looking at the x/100 in the chat screen
# if so, it dpes OCR
# TODO: either resize image to 1920 x 1080 , or detect the x/100 through another method
def CheckIfChatScreen(frame,frameCount = 0):
    # 1338 , 822     1402,849

    img = cv2.resize(frame, (1920, 1080))  # resize the image to (1920, 1080)

    crop_img = img[822:850, 1338:1410] #currently testing for 1920 x 1080
    #crop_img = cv2.bitwise_not(crop_img) #invert the image for Tesseract

    #cv2.imshow("crop_img", crop_img)
    #cv2.waitKey()
    cv2.imwrite("D:/Tesseract-OCR/test.png", crop_img)
    cmd = "D:/Tesseract-OCR/tesseract.exe  D:/Tesseract-OCR/test.png D:/Tesseract-OCR/test"
    os.system(cmd) #currently using os call for tesseract. TODO: use python tesseract library instead
    f = open("D:/Tesseract-OCR/test.txt", "r") #read the results
    OCRresult = f.read()
    if OCRresult != "":
      print(OCRresult) #print them
    if "100" in OCRresult:
        ExtractTextChatScreen(img,frameCount)
        #cv2.waitKey()



'''

        cv2.imshow("crop_img", crop_img)

        # convert to grayscale
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshhold
        ret, bin = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

        # closing
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)

        # invert black/white
        inv = cv2.bitwise_not(closing)
        cv2.imshow("img_outline", closing)
        cv2.waitKey()

'''

def ProcessVideo(location,outputlocation):


    #print(location)
    #clear previous data
    global ChatLogArray
    ChatLogArray = []


    cap = cv2.VideoCapture(location)

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 2200)

    currentFrame = 0
    #tickmeter = cv2.TickMeter()
    # while cv2.waitKey(1) < 0:
    while (cap.isOpened()):
        # print("next frame")
        # Read frame
        hasFrame, frame = cap.read()
        currentFrame = currentFrame + 1
        if not hasFrame:
            cv2.waitKey()
            break

        CheckIfChatScreen(frame,currentFrame)
        # GrabSettings(frame)
        # TextDetection(frame)

        # cv2.waitKey()
        # tickmeter.reset()

        # skip 5 frames
        # TODO: use grab() instead so OpenCV doesn't have to decode
        framesToSkip = 120
        while framesToSkip > 0:
            framesToSkip = framesToSkip - 1
            cap.read()
            currentFrame = currentFrame + 1

    #global ChatLogArray

    #f = open("D:/PythonCVtest/test.txt", "w")
    f = open(outputlocation, "w")
    print(ChatLogArray)

    totalIterations = 0

    f.write("{\n \"ChatMessages\": [")
    for chat in ChatLogArray:
        totalIterations = totalIterations + 1
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


        f.write("    \"COLORNAME\": \"")
        f.write(str(chat.colorName))
        f.write("\",\n")

        nameWrite = chat.name.replace("\n\x0c", "").replace("\n", "")
        nameWrite = "    \"NAME\": \"" + nameWrite + "\",\n"
        f.write(nameWrite)
        textWrite = chat.message.replace("\n\x0c", "").replace("\n", "")
        textWrite = "    \"MESSAGE\": \"" + textWrite + "\"\n"
        f.write(textWrite)
        if (totalIterations) == len(ChatLogArray):
            f.write("  }\n")
        else:
            f.write("  },\n")

    f.write("  ]\n}\n")

    f.close()
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def main():
  #grabcolors()

  # Read and store arguments
  confThreshold = 0.3 # Confidence threshold.
  nmsThreshold = 0.2 #Non-maximum suppression threshold.
  inpWidth = 1920 - 320  #resize  width (mutliple of 32)
  inpHeight = 1056 + 32 #resize  height (mutliple of 32)
  modelDetector = "D:/StellarisAI/frozen_east_text_detection.pb"  # Path to a binary .pb file contains trained detector network.'
  #modelRecognition = args.ocr

  # Load network
  global detector
  detector = cv2.dnn.readNet(modelDetector)

  # Open a video file or an image file or a camera stream
  #cap = cv2.VideoCapture("D:/PythonCVtest/chat.jpg")
  #cap = cv2.VideoCapture('D:/PythonCVtest/Videos/crewmate3longWin.mp4')
  #cap = cv2.VideoCapture('D:/PythonCVtest/Videos/harroTest.mkv')

  #glob.glob("/home/adam/*.mkv")
  videoLocation = 'D:/PythonCVtest/DataSets/Dataset_2_Among_Us/Crewmate/'
  videos = os.listdir(videoLocation)
  for video in videos:
      #outputLoc = videoLocation + video.replace(".mkv", ".txt")
      #f = open(outputLoc, "w+")
      #continue
      videoLoc = videoLocation + video
      outputLoc =  videoLocation + video.replace(".mkv",".json")
      ProcessVideo(videoLoc, outputLoc)
      print(video)
  exit(0)

  ProcessVideo('D:/PythonCVtest/Videos/harroTest.mkv',"D:/PythonCVtest/test.txt")

  exit(0)

  #cap.set(cv2.CAP_PROP_POS_FRAMES, 2200)

  tickmeter = cv2.TickMeter()
  #while cv2.waitKey(1) < 0:
  while (cap.isOpened()):
      #print("next frame")
      # Read frame
      hasFrame, frame = cap.read()
      if not hasFrame:
          cv2.waitKey()
          break

      CheckIfChatScreen(frame)
      #GrabSettings(frame)
      #TextDetection(frame)

      #cv2.waitKey()
      #tickmeter.reset()

      #skip 5 frames
      #TODO: use grab() instead so OpenCV doesn't have to decode
      framesToSkip = 120
      while framesToSkip > 0:
          framesToSkip = framesToSkip - 1
          cap.read()

  global ChatLogArray
  f = open("D:/PythonCVtest/test.txt", "w")
  print(ChatLogArray)

  totalIterations = 0

  f.write("{\n \"ChatMessages\": [")
  for chat in ChatLogArray:
    totalIterations = totalIterations + 1
    f.write("  {\n")
    f.write("    \"ID\": ")
    f.write( str(chat.id))
    f.write( ",\n")

    f.write("    \"RED\": ")
    f.write( str(chat.r))
    f.write(",\n")

    f.write("    \"GREEN\": ")
    f.write(str(chat.g))
    f.write(",\n")

    f.write("    \"BLUE\": ")
    f.write(str(chat.b))
    f.write(",\n")

    f.write("    \"COLORNAME\": ")
    f.write(str(chat.colorName))
    f.write(",\n")

    f.write("    \"FRAMECOUNT\": ")
    f.write(str(chat.frameCount))
    f.write(",\n")

    nameWrite = chat.name.replace("\n\x0c","").replace("\n","")
    nameWrite = "    \"NAME\": \"" + nameWrite + "\",\n"
    f.write(nameWrite)
    textWrite = chat.message.replace("\n\x0c", "").replace("\n","")
    textWrite = "    \"MESSAGE\": \"" + textWrite + "\"\n"
    f.write(textWrite)
    if (totalIterations) == len(ChatLogArray):
        f.write("  },\n")
    else:
      f.write("  },\n")

  f.write("  ]\n}\n")

  f.close()
  # When everything done, release the video capture object
  cap.release()
  # Closes all the frames
  cv2.destroyAllWindows()


  exit(0)

  '''

  cap = cv2.VideoCapture('D:/PythonCVtest/Videos/crewmate3longWin.mp4')
  # Check if camera opened successfully
  if (cap.isOpened() == False):
    print("Error opening video stream or file")
  # Read until video is completed
  ret, frame = cap.read()
  #cv2.imwrite("D:/PythonCVtest/Videos/testframe.png", frame)
  while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      # Display the resulting frame
      cv2.imshow('Frame', frame)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Break the loop
    else:
      break
  # When everything done, release the video capture object
  cap.release()
  # Closes all the frames
  cv2.destroyAllWindows()

'''

if __name__ == "__main__":
  main()



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
'''
cap = cv2.VideoCapture('D:/PythonCVtest/Videos/crewmate3longWin.mp4')
# Check if camera opened successfully
if (cap.isOpened() == False):
  print("Error opening video stream or file")
# Read until video is completed
ret, frame = cap.read()
cv2.imwrite("D:/PythonCVtest/Videos/testframe.png",frame)
while (cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

   # Break the loop
  else:
    break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
'''





# You can give path to the
# image as first argument
#img = cv2.imread('D:/PythonCVtest/stellaris.jpg')

# will show the image in a window
#cv2.imshow('image', img)
#k = cv2.waitKey(0) & 0xFF