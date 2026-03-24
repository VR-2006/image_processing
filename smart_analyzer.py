import cv2
import numpy as np

def analyze_image(path):
    img = cv2.imread(path)

    if img is None:
        print("Error: Image not found at the specified path.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) 

    kernel = np.ones((5,5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0   

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0), 2)
            count += 1

    print(f"Number of objects found: {count}")

    cv2.imwrite("analyzed_image.jpg", img)
    cv2.imwrite("processed_image.jpg", clean)

    return img, clean

#Input from user

path = input("Enter the path to the image: ")

result, processed = analyze_image(path)

result_small = cv2.resize(result, (600, 400))
processed_small = cv2.resize(processed, (600, 400))

cv2.imshow("Final Output", result_small)
cv2.imshow("Processed Image", processed_small)

cv2.waitKey(0)
cv2.destroyAllWindows()