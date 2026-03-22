import cv2

img = cv2.imread("mri.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

_, thresh = cv2.threshold(blur, 150,255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

cv2.imshow("Medical Image", img)
cv2.imshow("Thresholded",thresh)

cv2.waitKey(0)
cv2.destroyAllWindows() 

