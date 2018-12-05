import cv2
import numpy as np

video =cv2.VideoCapture("road_car_view.mp4")
vertices = np.array([[0,750],[1200,750],[1200,500],[800,450],[400,450],[0,500]], np.int32)
lside=np.array([[0,750],[600,750],[600,450],[400,450],[0,500]], np.int32)
rside=np.array([[600,750],[1200,750],[1200,500],[800,450],[600,400]], np.int32)

def roi(edges, vertices):
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, vertices, 255)
        masked = cv2.bitwise_and(edges, mask)
        return masked
    
def display_on_frame(self, image, left_curverad, right_curverad, car_off):
        """
        Display texts on image using passed values
        """
        font = cv2.FONT_HERSHEY_COMPLEX
        curve_disp_txt = 'Curvature: Right = ' + str(np.round(right_curverad,2)) + 'm, Left = ' + str(np.round(left_curverad,2)) + 'm' 

        off_disp_txt = 'Car off by ' + str(np.round(car_off,2)) + 'm'

        cv2.putText(image, curve_disp_txt, (30, 60), font, 1, (0,0,0), 2)
        cv2.putText(image, off_disp_txt, (30, 110), font, 1, (0,0,0), 2)
        
        return image

def mov_avg (mylist):
    N = 3
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)

    return moving_aves

def increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))


lwc=np.array([225,180,0])
hwc=np.array([255, 255, 170])


lyc=np.array([18,94,140])
hyc=np.array([48,255,255])

    
while True:
    ret, frame= video.read()
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    bin_thresh = [20, 255]
    
    mask1=cv2.inRange(hsv,lwc,hwc)
    mask2=cv2.inRange(hsv,lyc,hyc)
    
    #mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.addWeighted(mask1,0.5,mask2,0.5,0)
    #target = cv2.bitwise_and(frame,frame, mask=mask)

    edges=cv2.Canny(mask,75,150)
    
    processed_img1 = roi(edges, [lside])
    processed_img2 = roi(edges, [rside])
    processed_img =cv2.bitwise_or(processed_img1,processed_img2)
    processed_img=cv2.GaussianBlur(processed_img,(5,5),0)
        
    lines = cv2.HoughLinesP(processed_img,1,np.pi/180,250, maxLineGap=50, minLineLength=80)

    
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv2.line(frame,(x1,y1),(x2,y2),(0,250,0),5)

    cv2.line(frame,(600,750),(600,500),(255,0,255),5)

    para1 = cv2.HoughLines(processed_img1,1,np.pi/180,120)
    para2 = cv2.HoughLines(processed_img2,1,np.pi/180,120)
    p1=[]
    p2=[]
    
    if para1 is not None:
        for i in para1:
            cv2.putText(frame, 'left lane angle: '+str(round(i[0][1],3))+'rad' , (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), lineType=cv2.LINE_AA)
            p1.append(round(i[0][1]))
            #print (p1)
        if len(p1)>50:
            p1.pop(0)
    
    if para2 is not None:
        for j in para2:
            cv2.putText(frame, 'right lane angle: '+str(round(j[0][1],3))+'rad' , (750, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), lineType=cv2.LINE_AA)
            p2.append(round(i[0][1]))
        if len(p2)>50:
            p2.pop(0)

    if (para1 is None and para2 is None):
        cv2.putText(frame, 'no lanes detected' , (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), lineType=cv2.LINE_AA)
    elif para1 is not None and para2 is not None:
        if ((increasing(p1) and decreasing(p2)) or (increasing(p2) and decreasing(p1))):
            #print('t')
            cv2.putText(frame, 'driving towards a lane' , (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), lineType=cv2.LINE_AA)
    elif para1 is None:
        if (decreasing(p2)):
            #print('r')
            cv2.putText(frame, 'driving towards a lane' , (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), lineType=cv2.LINE_AA)
    elif para2 is None:
        if (decreasing(p1)):
            #print('l')
            cv2.putText(frame, 'driving towards a lane' , (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), lineType=cv2.LINE_AA)
    
    #r=cv2.selectROI(edges)
    #crop= edges[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
##    int minCannyThreshold = 190;
##    int maxCannyThreshold = 230;
##    Canny(LinesImg, LinesImg, minCannyThreshold, maxCannyThreshold, 5, true)
    
    #edges=cv2.Canny(gray,75,150)
    
    cv2.imshow('edges',edges)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(25)
    if key ==27:
        break
video.release()
cv2.destroyAllWindows()
