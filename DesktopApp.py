from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import os
import cv2
import numpy as np
import imutils
import time
import easygui


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()


        self.setWindowTitle("Document scaner")
        self.setGeometry(300, 750, 350, 300)

        #self.new_text = QtWidgets.QLabel(self)

        self.main_text = QtWidgets.QLabel(self)
        self.main_text.setText("Import image from: ")
        self.main_text.move(130, 50)
        self.main_text.adjustSize()

        self.btn1 = QtWidgets.QPushButton(self)
        self.btn1.move(70, 150)
        self.btn1.setText("Gallery")
        self.btn1.setFixedWidth(200)
        self.btn1.clicked.connect(self.btn_act1)

        self.btn2 = QtWidgets.QPushButton(self)
        self.btn2.move(70, 250)
        self.btn2.setText("Camera")
        self.btn2.setFixedWidth(200)
        self.btn2.clicked.connect(self.btn_act2)

        self.btn3 = QtWidgets.QPushButton(self)
        self.btn3.move(300, 5)
        self.btn3.setText("X")
        self.btn3.setFixedWidth(30)
        self.btn3.clicked.connect(self.btn_act3)

    # Gallery
    def btn_act1(self):
        print("Gallery")
        # set an original name based on date-time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        args_image = easygui.fileopenbox()
        # read image
        image = cv2.imread(args_image)
        orig = image.copy()

        # converting an image to grayscale. This will remove color noise.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # blur the picture to remove high-frequency noise
        # it helps to define the outline in the gray image
        grayImageBlur = cv2.blur(grayImage, (3, 3))

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
        ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02 * perimeter, True)

        # show outlines in image
        cv2.drawContours(image, [ROIdimensions], -1, (0, 255, 0), 2)

        # changing an array of coordinates
        ROIdimensions = ROIdimensions.reshape(4, 2)

        # ROI coordinate hold list
        rect = np.zeros((4, 2), dtype="float32")

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
        widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
        widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
        maxWidth = max(int(widthA), int(widthB))

        # calculate ROI height
        heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
        heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
        maxHeight = max(int(heightA), int(heightB))

        # a set of summary points for an overview of the entire document
        # new image size
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

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
        cv2.imwrite("images/output " + timestr + ".jpg", scanBW)


        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # # pop window
        # msg = QMessageBox()
        # msg.setWindowTitle("Info")
        # msg.setText("Scan saved!")
        # msg.setIcon(QMessageBox.Information)
        # msg.setStandartButtons(QMessageBox.Ok|QMessageBox.Cancel)
        #
        # msg.exec__()


    # camera
    def btn_act2(self):
        print("Camera")

        # set an original name based on date-time
        timestr = time.strftime("%Y%m%d-%H%M%S")
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
        # read image
        image = cv2.imread(args_image)
        orig = image.copy()

        # converting an image to grayscale. This will remove color noise.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # blur the picture to remove high-frequency noise
        # it helps to define the outline in the gray image
        grayImageBlur = cv2.blur(grayImage, (3, 3))

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
        ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02 * perimeter, True)

        # show outlines in image
        cv2.drawContours(image, [ROIdimensions], -1, (0, 255, 0), 2)

        # changing an array of coordinates
        ROIdimensions = ROIdimensions.reshape(4, 2)

        # ROI coordinate hold list
        rect = np.zeros((4, 2), dtype="float32")

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
        widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
        widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
        maxWidth = max(int(widthA), int(widthB))

        # calculate ROI height
        heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
        heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
        maxHeight = max(int(heightA), int(heightB))

        # a set of summary points for an overview of the entire document
        # new image size
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

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
        cv2.imwrite("images/output " + timestr + ".jpg", scanBW)
        # #pop window
        # msg = QMessageBox()
        # msg.exec_()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # exit
    def btn_act3(self):
        exit(0)


def appLication():
    app = QApplication(sys.argv)
    window = Window()

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    appLication()








# import cv2
# import numpy as np
# import imutils
# import time
# import easygui
#
# #Gallery
# def btn_act1():
#     #print("Gallery")
#
#
#     # set an original name based on date-time
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     args_image = easygui.fileopenbox()
#     # read image
#     image = cv2.imread(args_image)
#     orig = image.copy()
#
#     # converting an image to grayscale. This will remove color noise.
#     grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # blur the picture to remove high-frequency noise
#     # it helps to define the outline in the gray image
#     grayImageBlur = cv2.blur(grayImage, (3, 3))
#
#     # define the border using the Canny method
#     edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)
#
#     # find outlines in the cropped image, organize the area rationally
#     # leave only big options
#     allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     allContours = imutils.grab_contours(allContours)
#
#     # sorting the contours of the area by reduction and keeping the top-1
#     allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]
#
#     # contour approximation
#     perimeter = cv2.arcLength(allContours[0], True)
#     ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02 * perimeter, True)
#
#     # show outlines in image
#     cv2.drawContours(image, [ROIdimensions], -1, (0, 255, 0), 2)
#
#     # changing an array of coordinates
#     ROIdimensions = ROIdimensions.reshape(4, 2)
#
#     # ROI coordinate hold list
#     rect = np.zeros((4, 2), dtype="float32")
#
#     # the smallest amount will be at the top left corner,
#     # the largest - at the lower right corner
#     s = np.sum(ROIdimensions, axis=1)
#     rect[0] = ROIdimensions[np.argmin(s)]
#     rect[2] = ROIdimensions[np.argmax(s)]
#
#     # top-right will be with minimal difference
#     # bottom-left will have the maximum difference
#     diff = np.diff(ROIdimensions, axis=1)
#     rect[1] = ROIdimensions[np.argmin(diff)]
#     rect[3] = ROIdimensions[np.argmax(diff)]
#
#     # top-left, top-right, bottom-right, bottom-left
#     (tl, tr, br, bl) = rect
#
#     # calculate ROI width
#     widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
#     widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
#     maxWidth = max(int(widthA), int(widthB))
#
#     # calculate ROI height
#     heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
#     heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
#     maxHeight = max(int(heightA), int(heightB))
#
#     # a set of summary points for an overview of the entire document
#     # new image size
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")
#
#     # calculate the perspective transformation matrix and apply it
#     transformMatrix = cv2.getPerspectiveTransform(rect, dst)
#
#     # convert ROI
#     scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))
#
#     # convert to gray
#     scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
#
#     # converting to black and white with high contrast for documents
#     from skimage.filters import threshold_local
#
#     # increase contrast in the case of a document
#     T = threshold_local(scanGray, 9, offset=8, method="gaussian")
#     scanBW = (scanGray > T).astype("uint8") * 255
#
#     # show the final image with high contrast
#     cv2.imshow("Result", scanBW)
#     cv2.imwrite("images/output " + timestr + ".jpg", scanBW)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# #camera
# def btn_act2():
#     #print("Camera")
#
#
#     # set an original name based on date-time
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     # turning on a camera
#     cap = cv2.VideoCapture(0)
#     # "warm up" the camera so that the picture is not dark
#     for i in range(30):
#         cap.read()
#     # taking a picture
#     ret, frame = cap.read()
#
#     # write to file and
#     # set on original name for a photo from camera based on a current date and time
#     cv2.imwrite("images/input" + timestr + ".jpg", frame)
#     # turn off the camera
#     cap.release()
#
#     args_image = "images/input" + timestr + ".jpg"
#     # read image
#     image = cv2.imread(args_image)
#     orig = image.copy()
#
#     # converting an image to grayscale. This will remove color noise.
#     grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # blur the picture to remove high-frequency noise
#     # it helps to define the outline in the gray image
#     grayImageBlur = cv2.blur(grayImage, (3, 3))
#
#     # define the border using the Canny method
#     edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)
#
#     # find outlines in the cropped image, organize the area rationally
#     # leave only big options
#     allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     allContours = imutils.grab_contours(allContours)
#
#     # sorting the contours of the area by reduction and keeping the top-1
#     allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]
#
#     # contour approximation
#     perimeter = cv2.arcLength(allContours[0], True)
#     ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02 * perimeter, True)
#
#     # show outlines in image
#     cv2.drawContours(image, [ROIdimensions], -1, (0, 255, 0), 2)
#
#     # changing an array of coordinates
#     ROIdimensions = ROIdimensions.reshape(4, 2)
#
#     # ROI coordinate hold list
#     rect = np.zeros((4, 2), dtype="float32")
#
#     # the smallest amount will be at the top left corner,
#     # the largest - at the lower right corner
#     s = np.sum(ROIdimensions, axis=1)
#     rect[0] = ROIdimensions[np.argmin(s)]
#     rect[2] = ROIdimensions[np.argmax(s)]
#
#     # top-right will be with minimal difference
#     # bottom-left will have the maximum difference
#     diff = np.diff(ROIdimensions, axis=1)
#     rect[1] = ROIdimensions[np.argmin(diff)]
#     rect[3] = ROIdimensions[np.argmax(diff)]
#
#     # top-left, top-right, bottom-right, bottom-left
#     (tl, tr, br, bl) = rect
#
#     # calculate ROI width
#     widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
#     widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
#     maxWidth = max(int(widthA), int(widthB))
#
#     # calculate ROI height
#     heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
#     heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
#     maxHeight = max(int(heightA), int(heightB))
#
#     # a set of summary points for an overview of the entire document
#     # new image size
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")
#
#     # calculate the perspective transformation matrix and apply it
#     transformMatrix = cv2.getPerspectiveTransform(rect, dst)
#
#     # convert ROI
#     scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))
#
#     # convert to gray
#     scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
#
#     # converting to black and white with high contrast for documents
#     from skimage.filters import threshold_local
#
#     # increase contrast in the case of a document
#     T = threshold_local(scanGray, 9, offset=8, method="gaussian")
#     scanBW = (scanGray > T).astype("uint8") * 255
#
#     # show the final image with high contrast
#     cv2.imshow("Result", scanBW)
#     cv2.imwrite("images/output " + timestr + ".jpg", scanBW)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# #exit
# def btn_act3():
#     exit(0)
#
#
# def appLication():
#     app = QApplication(sys.argv)
#     window = QMainWindow()
#
#     window.setWindowTitle("Document scaner")
#     window.setGeometry(300, 750, 350, 300)
#
#     main_text = QtWidgets.QLabel(window)
#     main_text.setText("Import image from: ")
#     main_text.move(130, 50)
#     main_text.adjustSize()
#
#     btn1 = QtWidgets.QPushButton(window)
#     btn1.move(70, 150)
#     btn1.setText("Gallery")
#     btn1.setFixedWidth(200)
#     btn1.clicked.connect(btn_act1)
#
#     btn2 = QtWidgets.QPushButton(window)
#     btn2.move(70, 250)
#     btn2.setText("Camera")
#     btn2.setFixedWidth(200)
#     btn2.clicked.connect(btn_act2)
#
#     btn3 = QtWidgets.QPushButton(window)
#     btn3.move(300, 5)
#     btn3.setText("X")
#     btn3.setFixedWidth(30)
#     btn3.clicked.connect(btn_act3)
#
#     window.show()
#     sys.exit(app.exec_())
#
# if __name__ == "__main__":
#     appLication()