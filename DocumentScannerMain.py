import cv2
import numpy as np
import utlis

########################################################################
# Settings
webCamFeed = True
pathImage = "1.jpg"
cap = cv2.VideoCapture(1)
cap.set(10, 160)
heightImg = 640
widthImg = 480
########################################################################

utlis.initializeTrackbars()
count = 0

def preprocess_image(img):
    """
    Preprocess the input image: resize, grayscale, blur, and edge detection.
    """
    img = cv2.resize(img, (widthImg, heightImg))  # Resize image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Apply Gaussian blur
    return img, imgGray, imgBlur

def edge_detection(imgBlur):
    """
    Perform edge detection using Canny and morphological transformations.
    """
    thres = utlis.valTrackbars()  # Get threshold values from trackbars
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # Apply Canny edge detection
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # Apply dilation
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # Apply erosion
    return imgThreshold

def find_and_draw_contours(img, imgThreshold):
    """
    Find and draw contours on the image.
    """
    contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    imgContours = img.copy()
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # Draw all contours
    return imgContours, contours

def warp_perspective(img, biggest):
    """
    Warp the perspective to get a top-down view of the detected document.
    """
    pts1 = np.float32(biggest)  # Prepare points for warp
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # Prepare points for warp
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # Remove 20 pixels from each side
    imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
    imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
    return imgWarpColored

def adaptive_threshold(imgWarpColored):
    """
    Apply adaptive thresholding to the warped image.
    """
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
    imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
    imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)
    return imgWarpGray, imgAdaptiveThre

def display_result(imgArray, labels):
    """
    Stack and display images with labels.
    """
    stackedImage = utlis.stackImages(imgArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)

while True:
    if webCamFeed:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam")
            break
    else:
        img = cv2.imread(pathImage)
        if img is None:
            print(f"Failed to read image from {pathImage}")
            break

    img, imgGray, imgBlur = preprocess_image(img)
    imgThreshold = edge_detection(imgBlur)
    imgContours, contours = find_and_draw_contours(img, imgThreshold)

    # Find the biggest contour and warp perspective if found
    biggest, maxArea = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        imgBigContour = img.copy()
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # Draw the biggest contour
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        imgWarpColored = warp_perspective(img, biggest)
        imgWarpGray, imgAdaptiveThre = adaptive_threshold(imgWarpColored)

        # Image array for display
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
    else:
        imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Blank image
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    labels = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]]

    display_result(imageArray, labels)

    # Save image when 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"Scanned/myImage{count}.jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((stackedImage.shape[1] // 2) - 230, (stackedImage.shape[0] // 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", ((stackedImage.shape[1] // 2) - 200, (stackedImage.shape[0] // 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1

cap.release()
cv2.destroyAllWindows()
