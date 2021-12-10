import cv2.cv2
import cv2.cv2 as cv
import numpy as np

# 1 task
img = cv.imread('images/Lena.png')
img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

faces = cv.CascadeClassifier('faces.xml')
results = faces.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)

for (x, y, w, h) in results:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
cv.imwrite('search_face', img)

# 2 task
img = cv.imread('images/Lena.png')
img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

x1 = x - 0.1 * w
x2 = x + w
y1 = y - 0.1 * h
y2 = y + 1.14 * h
face_img = img[int(x1):int(x2), int(y1):int(y2)]
cv.imwrite('face.jpg', face_img)

# 3 task
img_2 = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
img_2 = cv.Canny(face_img, 100, 100)
cv.imwrite('binary.jpg', img_2)

# 4 task
final_contors = []
contours, _ = cv.findContours(img_2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for c in contours:
    x,y,w,h = cv.boundingRect(c)
    if w >= 10 and h >= 10:
        final_contors.append(c)
the_mask = np.zeros_like(img_2)
cv.drawContours(the_mask, final_contors, -1, (255, 255, 255), cv.FILLED)
img_3 = cv2.bitwise_and(img_2, img_2, mask=the_mask)
cv.imwrite('binary_second.jpg', img_3)

# 5 task
kernel = np.ones((5, 5), np.uint8)
img_4 = cv.dilate(img_3, kernel, 1)
cv.imwrite('morph.jpg', img_4)

# 6 task
img_5 = cv.GaussianBlur(img_4, (5, 5), 0)
cv.imwrite('gaus.jpg', img_5)
cv.normalize(img_5, np.zeros_like(img_5), 1, 0, cv.NORM_MINMAX)
cv.imwrite('gaus_with_norma.jpg', img_5)

# 7 task
img_6 = cv.bilateralFilter(face_img, 50, 50, cv.BORDER_REFLECT)
cv.imwrite('bilateral.jpg', img_6)

# 8 task
tmp = cv.cvtColor(face_img, cv.COLOR_BGR2LAB)
clahe = cv.createCLAHE(clipLimit=3., tileGridSize=(8, 8)) # CLAHE (Contrast Limited Adaptive Histogram Equalization)

l, a, b = cv.split(tmp)
l2 = clahe.apply(l)
tmp = cv.merge((l2, a, b)) # merge channels

img_7 = cv.cvtColor(tmp, cv.COLOR_LAB2BGR)
cv.imwrite('contrast.jpg', img_7)

# 9 task
gaus_val = np.expand_dims(img_5, axis=2)
img_8 = gaus_val * img_7 + (1 - gaus_val) * img_6
cv.imwrite("result.jpg", img_8)

cv.waitKey(0)
