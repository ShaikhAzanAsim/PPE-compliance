import cv2
import torch
from ultralytics import YOLO

# ===================== Load Models =====================
helmet_model = YOLO("helmet.pt")  # Your custom helmet detection model
model_path = r"C:\Users\QBS PC\QBS_CO\PoseEstimation\yolov8n-pose.pt"
pose_model = YOLO(model_path)     # YOLO pose model for human keypoints

# Force both models to use CPU
helmet_model.to("cpu")
pose_model.to("cpu")

# ===================== Utility Functions =====================
def boxes_overlap(box, keypoint, margin=10):
    """
    Check if helmet box overlaps with head keypoint.
    box = [x1, y1, x2, y2]
    keypoint = (x, y)
    margin = radius (tolerance around keypoint)
    """
    x1, y1, x2, y2 = box
    kx, ky = keypoint
    return (x1 <= kx <= x2) and (y1 <= ky <= y2)

# ===================== Video Inference =====================
video_path = r"C:\Users\QBS PC\QBS_CO\DEMO\PersonFallingDemo\fall.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video or webcam.")
    exit()

# --------------------- Setup Video Writer ---------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:  # fallback if not available
    fps = 25.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("helmet_inference_output.mp4", fourcc, fps, (width, height))
print("Saving output video to 'helmet_inference_output.mp4'")

# --------------------- Inference Loop ---------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Step 1: Helmet Detection ----
    helmet_results = helmet_model(frame, verbose=False)
    helmet_boxes = []
    for r in helmet_results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            helmet_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, "Helmet", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---- Step 2: Pose Detection ----
    pose_results = pose_model(frame, verbose=False)
    status_text = "No person detected"
    helmet_found = len(helmet_boxes) > 0

    for r in pose_results:
        for kps in r.keypoints.xy:
            # Keypoints: nose(0), eyes(1,2), ears(3,4)
            head_points = kps[0:5]  # Take top 5 for head region
            head_x = float(torch.mean(head_points[:, 0]))
            head_y = float(torch.mean(head_points[:, 1]))
            head_point = (head_x, head_y)

            cv2.circle(frame, (int(head_x), int(head_y)), 5, (255, 0, 0), -1)

            # ---- Step 3: Check Overlap ----
            helmet_on = any(boxes_overlap(hbox, head_point) for hbox in helmet_boxes)

            if helmet_on:
                status_text = "Helmet ON"
                color = (0, 255, 0)
            elif helmet_found:
                status_text = "Helmet OFF"
                color = (0, 0, 255)
            else:
                status_text = "No Helmet Found"
                color = (0, 255, 255)

            # Draw person head keypoint & status
            cv2.putText(frame, status_text, (int(head_x) - 50, int(head_y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ---- Step 4: Show & Save Output ----
    cv2.imshow("Helmet Detection", frame)
    out.write(frame)  # <--- save the current frame

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # <--- release video writer
cv2.destroyAllWindows()
