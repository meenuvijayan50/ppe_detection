import cv2
import time
from ultralytics import YOLO

model = YOLO("best.onnx")

# Video path
video_path = "ppe.mp4"

cap = cv2.VideoCapture(video_path)

# Optional: save output video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "output_detected.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video,
    (width, height)
)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # preprocess timing
    pre_start = time.time()

    img = frame.copy()

    pre_ms = (time.time() - pre_start) * 1000

    # inference timing
    infer_start = time.time()

    results = model(img)

    infer_time = time.time() - infer_start

    fps = 1 / infer_time if infer_time > 0 else 0

    # postprocess timing
    post_start = time.time()

    annotated = results[0].plot()

    post_ms = (time.time() - post_start) * 1000

    # overlay metrics
    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated,
        f"Pre: {pre_ms:.1f} ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        annotated,
        f"Post: {post_ms:.1f} ms",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    # save output
    out.write(annotated)

    # show
    cv2.imshow("Video Inference", annotated)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()