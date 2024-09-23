import cv2
import numpy as np
import math
import copy

def calculate_line_length(point 1, point2):
    x1, y1 = point1
    x2, y2 = point2

    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return length


def get_length_width(cnt):
    l1 = calculate_line_length(cnt[0], cnt[1])
    l2 = calculate_line_length(cnt[1], cnt[2])
    return min(l1, l2), max(l1, l2)


def compute_rotated_rectangle(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    extremes = rect[1]
    return box, extremes


def is_contour_inside(outer_contour, inner_contour, threshold=145):
    outer_contour_rect, sourceExtremes = compute_rotated_rectangle(outer_contour)
    inner_contour_rect, targetExtremes = compute_rotated_rectangle(inner_contour)


    outer_area = cv2.contourArea(outer_contour)
    inner_area = cv2.contourArea(inner_contour)

    outer_area_rect = cv2.contourArea(outer_contour_rect)
    inner_area_rect = cv2.contourArea(inner_contour_rect)

    print("outer_area " + str(outer_area))
    print("inner_area " + str(inner_area))


    if inner_area - outer_area > threshold:
        return False

    if inner_area_rect - outer_area_rect > threshold:
        return False

    inner_w, inner_h = get_length_width(outer_contour_rect)
    outer_w, outer_h = get_length_width(inner_contour_rect)
    if inner_w - outer_w > threshold or inner_h - outer_h > threshold:
        return False

    return True

#camera
cam = cv2.VideoCapture(0)
cam.set(3, 1920 / 4)
cam.set(4, 1080 / 4)
img_counter = 0
usingCamera = True
path = "formerot.png"

while True:
    ret, img = cam.read()
    img_with_text = copy.deepcopy(img)
    img_with_text = cv2.putText(img_with_text, "ESC - to close", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 0), 3, cv2.LINE_AA)
    img_with_text = cv2.putText(img_with_text, "ENTER - to capture picture", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("Webcam", img_with_text)

    key = cv2.waitKey(1)
    if key == 27:
        print("Closing app.")
        cam.release()
        cv2.destroyAllWindows()
        break
    elif key == 13:
        img_name = "frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, img)
        print("Picture captured!")
        img_counter += 1
        if usingCamera:
            path = img_name

        cam.release()
        cv2.destroyAllWindows()
        break





threshold = 125
img = cv2.imread(path)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 30, 200)
_, imgThre = cv2.threshold(imgCanny, 10, 255, 0);

contours, hierarchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for idx, cnt in enumerate(contours):
    box, ext = compute_rotated_rectangle(cnt)
    # draw minimum area rectangle (rotated rectangle)
    img = cv2.drawContours(img, [box], 0, (0, 255, 255), 2)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.putText(img, str(idx),
                (x, y),  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX,  # font family
                1,  # font size
                (209, 80, 0, 255),  # font color
                3)  # font stroke)

    for i in range(len(contours)):
        if i != idx:
            cntTarget = contours[i]
            #boxSource, sourceExtremes = compute_rotated_rectangle(cnt)
            #boxTarget, targetExtremes = compute_rotated_rectangle(cntComp)

            print('id outer:' + str(idx))
            print('id inner: ' + str(i))
            if is_contour_inside(cnt, cntTarget,threshold=threshold):
                cv2.putText(img, "(" + str(idx) + "," + str(i) + ")", (x, y + 15 * (i + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX,  # font family
                            0.5,  # font size
                            (209, 80, 0, 255),  # font color
                            2)  # font stroke))

cv2.imshow("Bounding Rectangles", img)
cv2.waitKey(0)
cam.release()
cv2.destroyAllWindows()
