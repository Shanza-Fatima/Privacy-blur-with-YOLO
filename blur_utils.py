#one function that blurs the rectangle of a     picture
import cv2
def blur_box(frame, x1, y1, x2, y2, ksize=51):
    height , width = frame.shape[:2] #frame shape is (rows, columns, BGR) we need only 2
    x1, y1 = max(0, x1), max(0,y1)
    x2,y2 = min(width, x2), min(height,y2)
    if x2 <= x1 or y2 <= y1:
        return
    if ksize %2 == 0:
#to find the center pixel if the given is in even we need to make it odd
        ksize += 1 
#cutout the rectange i,e [rows, columns]
    region = frame[y1:y2 , x1:x2]
#blur that piece and paste it back ,(ksize, ksize) => blur strength
    frame[y1:y2 , x1:x2] = cv2.GaussianBlur(region, (ksize, ksize), 0)
    