import cv2
import numpy as np

def blend_video_optimized(input_path, output_path="output.mp4"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("❌ Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎞 Processing {total_frames} frames at {fps:.2f} FPS ({width}x{height})")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    avg_frame = None
    alpha = 0  # adaptive blending factor

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame.astype(np.float32)

        if avg_frame is None:
            avg_frame = frame
        else:
            # Dynamically decrease alpha so older frames keep more influence
            alpha = 1.0 / (frame_idx + 1)
            cv2.accumulateWeighted(frame, avg_frame, alpha)

        blended = cv2.convertScaleAbs(avg_frame)
        out.write(blended)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Saved blended video as '{output_path}'")

# Example usage:
blend_video_optimized("input.mp4", "output.mp4")
