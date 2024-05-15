import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)


model_base_path = 'models/'
model_path = 'onnx_safefall.onnx'
video_cam_index = "videos/demo.mp4"

maskModel = cv2.dnn.readNetFromONNX(model_base_path + model_path)
stream = cv2.VideoCapture(video_cam_index)
output = ""
color = [(0, 0, 255), (0, 255, 0)]


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

        # Return bounding box coordinates and cropped image
        return min_x, min_y, max_x, max_y, cropped_img

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
    while True:
        # Read frame from the stream
        ret, frame = stream.read()

        # Call the pose detection function to get bounding box coordinates and cropped image
        pose_result = detect_pose(frame)

        if pose_result:
            min_x, min_y, max_x, max_y, cropped_img = pose_result

            # Run the detect function on the cropped image
            predictions = detect(cropped_img, maskModel)

            if predictions == 1:
                output = "Patient is safe"
            elif predictions == 0:
                output = "Patient has fallen, alerting nurses..."
            else:
                pass

            frame = cv2.putText(frame, output, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color[predictions], 3, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Break from loop if key pressed is q
        if key == ord("q"):
            break


if __name__ == "__main__":
    safefall()