import cv2
import numpy as np
import imutils
import time
import easygui



#set an original name based on date-time
timestr = time.strftime("%Y%m%d-%H%M%S")

#menu
print("Choose a source: ")
choice ='0'
while choice =='0':
    print("Camera \t-> \tpress 1")
    print("Gallery -> \tpress 2")
    print("Exit \t-> \tpress 0")

    choice = input ("Input: ")

    if choice == "0":
        print("Exit")
        exit(0)
    elif choice == "1":
        # turning on a camera
        cap = cv2.VideoCapture(0)
        # "warm up" the camera so that the picture is not dark
        for i in range(30):
            cap.read()
        # taking a picture
        ret, frame = cap.read()

        # write to file and
        # set on original name for a photo from camera based on a current date and time
        cv2.imwrite("images/input" + timestr + ".jpg", frame)
        # turn off the camera
        cap.release()

        args_image = "images/input" + timestr + ".jpg"

    elif choice == "2":
        args_image = easygui.fileopenbox()

    else:
        print("Out of range")


# read image
image = cv2.imread(args_image)
orig = image.copy()

# converting an image to grayscale. This will remove color noise.
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur the picture to remove high-frequency noise
# it helps to define the outline in the gray image
grayImageBlur = cv2.blur(grayImage,(3,3))

# define the border using the Canny method
edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)

# find outlines in the cropped image, organize the area rationally
# leave only big options
allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
allContours = imutils.grab_contours(allContours)

# sorting the contours of the area by reduction and keeping the top-1
allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]

# contour approximation
perimeter = cv2.arcLength(allContours[0], True)
ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)

# show outlines in image
cv2.drawContours(image, [ROIdimensions], -1, (0,255,0), 2)

# changing an array of coordinates
ROIdimensions = ROIdimensions.reshape(4,2)

# ROI coordinate hold list
rect = np.zeros((4,2), dtype="float32")

# the smallest amount will be at the top left corner,
# the largest - at the lower right corner
s = np.sum(ROIdimensions, axis=1)
rect[0] = ROIdimensions[np.argmin(s)]
rect[2] = ROIdimensions[np.argmax(s)]

# top-right will be with minimal difference
# bottom-left will have the maximum difference
diff = np.diff(ROIdimensions, axis=1)
rect[1] = ROIdimensions[np.argmin(diff)]
rect[3] = ROIdimensions[np.argmax(diff)]

# top-left, top-right, bottom-right, bottom-left
(tl, tr, br, bl) = rect

# calculate ROI width
widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
maxWidth = max(int(widthA), int(widthB))

# calculate ROI height
heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
maxHeight = max(int(heightA), int(heightB))

# a set of summary points for an overview of the entire document
# new image size
dst = np.array([
    [0,0],
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight-1],
    [0, maxHeight-1]], dtype="float32")

# calculate the perspective transformation matrix and apply it
transformMatrix = cv2.getPerspectiveTransform(rect, dst)

# convert ROI
scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))

# convert to gray
scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

# converting to black and white with high contrast for documents
from skimage.filters import threshold_local

# increase contrast in the case of a document
T = threshold_local(scanGray, 9, offset=8, method="gaussian")
scanBW = (scanGray > T).astype("uint8") * 255

# show the final image with high contrast
cv2.imshow("Result", scanBW)
cv2.imwrite("images/output " +timestr+ ".jpg", scanBW)
cv2.waitKey(0)
cv2.destroyAllWindows()