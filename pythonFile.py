import cv2
import cvzone
#faceDectionMdule has 6 land marks, FaceMeshDetector 468 landmarks for more accurate results
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

faceID = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

cap = cv2.VideoCapture(0)
#only detect one face
detector = FaceMeshDetector(maxFaces=1)
#defining the plot
chartEye = LivePlot(640,480,[20,40])

# loop runs if capturing has been initialized. 
while 1: 

    # need the following three commands to show the camera view in live
    #ret, image1 = cap.read() 
    #cv2.imshow('Live View',image1)
    #cv2.waitKey(10)


    ret, image1 = cap.read() 
    #this line draws a meshmap on the face, draw = true ot false determines if we want to plot the mesh on the display
    image1, faces = detector.findFaceMesh(image1, draw=False)

    #if the list faces contains something
    if faces:
        #we pickout the first face from the list, in this case we only have one face
        firstFace = faces[0]
        
        #//this for loop is used to print the mesh cirlces around the eye
        #for id in faceID:
            #cv2.circle(image1, firstFace[id], 1, (0,200,0), cv2.FILLED) 
        
        
        #gathering parameters points
        leftTop = firstFace[159]
        leftBottom = firstFace[23]
        leftLeft = firstFace[130]
        leftRight = firstFace[243]

        #gathering center points of the left and right eye
        rightCenter = firstFace[145]
        leftCenter = firstFace[374]
        #cv2.line(image1,rightCenter,leftCenter,(0,200,0),1)      
        eyeDistance,_ = detector.findDistance(rightCenter,leftCenter)
        print(eyeDistance)

        #the average distance between the eyes is 6.3cm, and 50cm is the distance from the camera
        #the focal length is calculated to be 830
        width = 6.3
        distance = 50
        focalLength = 830

        distance1 = (width*focalLength)/eyeDistance
        #print(distance1)
        cvzone.putTextRect(image1, f'Dist: {int(distance1)} cm', (firstFace[10][0]+50, firstFace[10][1]-50), scale=1.8)
        
        #finding the distance between these points, if you add ,_ it does not print extra info
        ratio1,_ = detector.findDistance(leftTop, leftBottom)
        disVertical,_ = detector.findDistance(leftTop, leftBottom)
        disHorizontal,_ = detector.findDistance(leftLeft, leftRight)
        ratio = (disVertical/disHorizontal)*100
        #print(ratio)

        #drawing the lines on our live plot
        #cv2.line(image1,leftTop,leftBottom,(200,200,200),4)      
        #cv2.line(image1,leftLeft,leftRight,(200,0,200),4)

        #open eyes the ratio is around 23, closed eyes around 35
        if ratio < 30:
            cvzone.putTextRect(image1, "Sleepy", (firstFace[10][0]+50, firstFace[10][1]), scale=1.8, colorT= (255, 0, 0))
            #cv2.putText(image1, "Sleepy", leftTop, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2 )
        else:
            cvzone.putTextRect(image1, "Awake", (firstFace[10][0]+50, firstFace[10][1]), scale=1.8)
            #cv2.putText(image1, "Awake", leftTop, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2 )


        #the chart plot needs to be in a variable 
        chartPlot = chartEye.update(ratio)
        #cv2.imshow("Update graph", chartPlot)
        
        imageStack = cvzone.stackImages([image1,chartPlot],1, 1)

    #image2 = cv2.resize(image1, (640,360))
     
    cv2.imshow('Live View',imageStack)
    cv2.waitKey(10)


# Close the window 
cap.release() 
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  

