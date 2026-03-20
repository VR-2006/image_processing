import cv2
img = cv2.imread("input.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original", img)
cv2.imshow("Grayscale",gray)

print(gray.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()