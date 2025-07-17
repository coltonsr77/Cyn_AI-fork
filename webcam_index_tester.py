import cv2

def test_camera_until_valid_frame(index=1, target_frame=35, max_frames=36):
    print(f"Opening camera at index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Failed to open camera at index {index}")
        return

    valid_frame = None
    frame_count = 0

    print(f"Capturing up to {max_frames} frames to find a valid one...")

    while frame_count < max_frames:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print(f"Frame {frame_count}: Failed to capture")
            continue

        print(f"Frame {frame_count}: Captured")

        # Check if this is the target frame to show
        if frame_count == target_frame:
            valid_frame = frame
            break

    cap.release()

    if valid_frame is not None:
        window_name = f"Valid Frame #{target_frame}"
        cv2.imshow(window_name, valid_frame)
        print(f"Showing frame #{target_frame}. Press 'q' or close window to exit.")

        while True:
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # If window is closed by user, break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
    else:
        print(f"No valid frame found at or before frame {max_frames}.")

if __name__ == "__main__":
    test_camera_until_valid_frame()
