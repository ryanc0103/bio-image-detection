import cv2
import numpy as np

def enhance_image(gray, method="clahe"):
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)
    return gray


def enhance_image(gray: np.ndarray, method: str = "clahe") -> np.ndarray:
    """
    Enhance a grayscale image to help detection on low-quality inputs.
    method: "clahe", "median", "sharpen", "denoise", or "none"
    """
    if method == "none":
        return gray

    if method == "clahe":
        # (clahe - contrast limited adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    if method == "median":
        return cv2.medianBlur(gray, 5)

    if method == "denoise":
        # denoising for grayscale images
        return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    if method == "sharpen":
        # sharpening kernel
        kernel = np.array([[0, -1,  0],
                           [-1, 5, -1],
                           [0, -1,  0]], dtype=np.float32)
        return cv2.filter2D(gray, -1, kernel)

    raise ValueError(f"Unknown enhancement method: {method}")


# load image
image = cv2.imread("images/image1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# enhancement
gray_enh = enhance_image(gray, method="clahe") 

# continued pipeline for enhancement
blur = cv2.GaussianBlur(gray_enh, (5, 5), 0)
_, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


clahe = cv2.createCLAHE(clipLimit=2.0)
enhanced = clahe.apply(gray)

cv2.imshow("Gray Original", gray)
cv2.imshow("Gray Enhanced", gray_enh)

# blur, reduce noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# threshold
_, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

# cleanup
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# find contours
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:  # filter noise
        count += 1
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

print(f"Detected objects: {count}")

cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
