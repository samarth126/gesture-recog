import cv2

cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame")
        break
    

    print(frame)
    print(frame.shape)
    
    break
    cv2.imshow("Video", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()