import cv2
import numpy as np

img = cv2.imread("mri.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

_, thresh = cv2.threshold(blur, 120,255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

cv2.imwrite("output_detected.jpeg", img)
cv2.imwrite("output_thresh.jpeg", clean)

cv2.imshow("Final Output", img)
cv2.imshow("Processed", clean)

cv2.waitKey(0)
cv2.destroyAllWindows() 