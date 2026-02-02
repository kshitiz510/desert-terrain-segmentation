import cv2, numpy as np, os
p = cv2.imread("results/test_predictions/" + os.listdir("results/test_predictions")[0], cv2.IMREAD_GRAYSCALE)
print("Unique predicted labels:", np.unique(p))