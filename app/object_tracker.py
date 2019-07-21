import numpy as np
import cv2


class ColourTracker:
    def __init__(self):
        cv2.namedWindow("Background")
        cv2.namedWindow("Frame")

        self.capture = cv2.VideoCapture(0)
        self.knn = cv2.createBackgroundSubtractorKNN()

    def run(self):
        while True:
            f, orig_img = self.capture.read()

            if not f:
                break

            fore = self.knn.apply(orig_img)
            back = self.knn.getBackgroundImage()
            kernel = np.ones((5, 5), np.uint8)
            fore = cv2.erode(fore, kernel)
            fore = cv2.dilate(fore, kernel)
            # fore = fore[r:r+h, c:c+w]
            image, contours = cv2.findContours(
                fore, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            maximumArea = 0
            bestContour = None
            for contour in contours:
                currentArea = cv2.contourArea(contour)
                if currentArea > maximumArea:
                    bestContour = contour
                    maximumArea = currentArea

                    x, y, w, h = cv2.boundingRect(bestContour)
                    cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

            cv2.imshow("Background", back)
            cv2.imshow("Frame", orig_img)

            k = cv2.waitKey(24) & 0xFF
            print(k)
            if k == 27:
                break


if __name__ == "__main__":
    colour_tracker = ColourTracker()
    colour_tracker.run()
