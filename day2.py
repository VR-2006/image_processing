import cv2
img = cv2.imread("input.jpg")

blur = cv2.GaussianBlur(img, (15,15), 0)

cv2.imshow("Original", img)
cv2.imshow("Blurred", blur) 

cv2.waitKey(0)
cv2.destroyAllWindows()