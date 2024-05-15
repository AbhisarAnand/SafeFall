import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False)

video_name = "normal_patient.mp4"
cap = cv2.VideoCapture(video_name)

extra_width = 50  
extra_height = 60 

output_dir = "images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = pose_estimator.process(frame)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), -float('inf'), -float('inf')
        for landmark in pose_landmarks:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        box_width = max_x - min_x
        box_height = max_y - min_y

        center_x = min_x + box_width // 2
        center_y = min_y + box_height // 2

        box_width += 2 * extra_width
        box_height += 2 * extra_height

        box_width = min(frame.shape[1] - center_x, center_x, box_width)
        box_height = min(frame.shape[0] - center_y, center_y, box_height)

        cv2.rectangle(frame, (center_x - box_width // 2, center_y - box_height // 2),
                      (center_x + box_width // 2, center_y + box_height // 2), (255, 0, 0), 2)

        cropped_img = frame[center_y - box_height // 2:center_y + box_height // 2,
                            center_x - box_width // 2:center_x + box_width // 2]

        image_counter += 1
        image_name = f"{output_dir}/{video_name.split(".")[0]}_{image_counter}.jpg"
        cv2.imwrite(image_name, cropped_img)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
