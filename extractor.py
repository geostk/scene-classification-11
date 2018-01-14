import cv2
import os

EXAMPLE_IMG = os.path.join("images","101_ObjectCategories","pyramid","image_0056.jpg")

def sift_features(dir_img):
    input_img = cv2.imread(dir_img)
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kps, descs = sift.detectAndCompute(gray, None)
    print("kps: ",kps)
    print("desc: ",descs)
    print("len desc: ",len(descs))

    return kps, descs

sift_features(EXAMPLE_IMG)
