import cv2
import mediapipe as mp
import os

# Initialize the MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Create a video capture object
video_full_name = "fallen.mp4"
video_name = video_full_name.split(".")[0]
cap = cv2.VideoCapture(video_full_name)

# Directory to save cropped images
output_dir = f"images/{video_name}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Counter for saved images
image_counter = 0

# Fixed increase in bounding box size
extra_width = 50
extra_height = 100

# Loop over the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the MediaPipe Pose model on the image
    results = pose.process(image)

    # Draw the pose landmarks on the image
    if results.pose_landmarks:
        # Draw pose landmarks and connections
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract pose landmarks
        pose_landmarks = results.pose_landmarks.landmark
        
        # Find bounding box coordinates
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), -float('inf'), -float('inf')
        for landmark in pose_landmarks:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        # Increase bounding box size
        min_x -= extra_width
        min_y -= extra_height
        max_x += extra_width
        max_y += extra_height


        # Make sure the bounding box coordinates are within the frame
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(frame.shape[1], max_x)
        max_y = min(frame.shape[0], max_y)

        # Draw bounding box around the entire person
        cv2.rectangle(annotated_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)

        # Crop the bounding box area
        cropped_img = frame[min_y:max_y, min_x:max_x]

        # Save the cropped image
        image_counter += 1
        image_name = f"{output_dir}/{video_name}_{image_counter}.jpg"
        cv2.imwrite(image_name, cropped_img)

        # Display the annotated image
        cv2.imshow('MediaPipe Pose with Bounding Box', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
