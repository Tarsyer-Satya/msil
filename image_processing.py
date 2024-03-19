import cv2
import numpy as np
from helper_functions import *


# points = np.array([[451,683],[641,356],[370,0],[63,371]], np.int32).reshape((-1,1,2))
# obj = Find_Bolts('frame5.jpg', roi = points)
# obj.apply_effects(blur = True, smooth= True, threshold = True, erosion = True, dilation =True)
# print(obj.find_contours())
        


# points_for_cards =  np.array([[913,715],[1541,868],[1667,68],[1196,16]], np.int32).reshape((-1,1,2))
# card_obj = Find_Boxes('frame2.jpg',roi = points_for_cards, inv_mask = True)
# card_obj.apply_effects(threshold = True, mask_inv= True)
# print(card_obj.find_contours())
        

# points_for_cards =  np.array([[510,672],[630,765],[751,546],[653,413]], np.int32).reshape((-1,1,2))
# card_obj = Gourmet_Fitment('frame7.jpg',roi = points_for_cards, inv_mask = True)
# card_obj.apply_effects(threshold = True, dilation = True, mask_inv= True)
# print(card_obj.find_contours())


import cv2

# Open the video file
video_path = 'video_2024-02-16_12-49-17.h264'
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'processed_output_video2.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

skip_frame = 5
frame_count = 0


# Process each frame of the video   
while cap.isOpened():
    # Read a frame from the video
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if end of video is reached

    # if frame_count % skip_frame != 0:
    #     continue

    # points = np.array([[451,683],[641,356],[370,0],[63,371]], np.int32).reshape((-1,1,2))
    # obj = Find_Bolts(image = frame, roi = points)
    # obj.apply_effects(blur = True, smooth= True, threshold = True, erosion = True, dilation =True)
    # print(obj.find_contours())


    points_for_cards =  np.array([[913,715],[1541 -25,868],[1667-25,68],[1196,16]], np.int32).reshape((-1,1,2))
    card_obj = Find_Boxes(image = frame,roi = points_for_cards, inv_mask = True)
    card_obj.apply_effects(threshold = True, mask_inv= True)
    card_obj_res = card_obj.find_contours()
    # print(card_obj.find_contours())

    points_for_fitment =  np.array([[510,672],[630,765],[751,546],[653,413]], np.int32).reshape((-1,1,2))
    fitment_obj = Gourmet_Fitment('frame7.jpg',roi = points_for_fitment, inv_mask = True)
    fitment_obj.apply_effects(threshold = True, dilation = True, mask_inv= True)
    fitment_obj_res = fitment_obj.find_contours()
    # print(fitment_obj.find_contours())



    
    # Display the frame
    cv2.imshow('Frame', putText([f"cards-{card_obj_res['card_boxes_count']}", f"Boxes-{card_obj_res['boxes_count']}", f"Gourmet-{fitment_obj_res['Number_of_Fitments']}"],img = frame))
    
    # Write the frame to the output video file
    out.write(frame)

    # Wait for 25 milliseconds and check if the 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

# print(f"Video processing complete. Output saved as {output_path}.")


