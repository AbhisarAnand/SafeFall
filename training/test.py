import cv2
import mediapipe as mp
import numpy as np
from collections import Counter

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)

model_base_path = 'models/'
model_path = 'onnx_safefall.onnx'
video_cam_index = "videos/demo.mp4"
output_video_path = "videos/demo_output.mp4"

maskModel = cv2.dnn.readNetFromONNX(model_base_path + model_path)
stream = cv2.VideoCapture(video_cam_index)
output = ""
color = [(0, 0, 255), (0, 255, 0)]
most_common_prediction = 0


def detect_pose(frame):
    # Run the MediaPipe Pose model on the frame
    results = pose.process(frame)

    # Extract pose landmarks
    if results.pose_landmarks:
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
        extra_width = 50
        extra_height = 100
        min_x -= extra_width
        min_y -= extra_height
        max_x += extra_width
        max_y += extra_height

        # Make sure the bounding box coordinates are within the frame
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(frame.shape[1], max_x)
        max_y = min(frame.shape[0], max_y)

        # Crop the bounding box area
        cropped_img = frame[min_y:max_y, min_x:max_x]

        # Adjust pose landmarks coordinates to match cropped region
        for landmark in pose_landmarks:
            landmark.x = (landmark.x * frame.shape[1] - min_x) / cropped_img.shape[1]
            landmark.y = (landmark.y * frame.shape[0] - min_y) / cropped_img.shape[0]

        # Return bounding box coordinates and cropped image
        return min_x, min_y, max_x, max_y, cropped_img, results.pose_landmarks

    else:
        return None


def detect(img, mask_model=maskModel):
    # Generate a blob
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (128, 128), (0, 0, 0), swapRB=True, crop=False)

    mask_model.setInput(blob)
    preds = mask_model.forward()
    prediction_index = np.array(preds)[0].argmax()
    return prediction_index


def safefall():
    global maskModel, output
    global stream
    global most_common_prediction
    predictions_list = []
    frame_counter = 0

    # Get video properties
    fps = int(stream.get(cv2.CAP_PROP_FPS))
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        # Read frame from the stream
        ret, frame = stream.read()
        if not ret:
            break

        frame_counter += 1

        # Call the pose detection function to get bounding box coordinates and cropped image
        pose_result = detect_pose(frame)

        if pose_result:
            min_x, min_y, max_x, max_y, cropped_img, pose_landmarks = pose_result

            # Draw pose landmarks on the cropped image
            mp_drawing.draw_landmarks(cropped_img, pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw bounding box around the person
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

            # Run the detect function on the cropped image
            predictions = detect(cropped_img, maskModel)
            predictions_list.append(predictions)
            if frame_counter % 3 == 0:
                most_common_prediction = Counter(predictions_list).most_common(1)[0][0]

                if most_common_prediction == 1:
                    output = "Patient is safe"
                elif most_common_prediction == 0:
                    output = "Patient has fallen, activating CODE BLUE..."
                else:
                    pass

                predictions_list = []
            frame = cv2.putText(frame, output, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color[most_common_prediction], 3, cv2.LINE_AA)

        # Save frame to output video
        out.write(frame)

        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Break from loop if key pressed is q
        if key == ord("q"):
            break

    # Release video writer and capture objects
    out.release()
    stream.release()

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    safefall()
