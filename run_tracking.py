# run_tracking.py
import os
import cv2
import argparse
import sys
from object_tracker import ObjectTracker # Imports the modified class

def parse_args():
    parser = argparse.ArgumentParser(description='Object Tracking with YOLOv8 (ONNX) and DeepSORT')
    # --- Changed model argument ---
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model file (.onnx)')
    # --- Input arguments ---
    parser.add_argument('--video', type=str, default='', help='Path to input video file')
    parser.add_argument('--image_dir', type=str, default='', help='Path to directory with images (processed sequentially)')
    parser.add_argument('--camera_id', type=int, default=-1, help='Camera ID for live tracking (e.g., 0)')
    # --- Output arguments ---
    parser.add_argument('--output_dir', type=str, default='output_tracking', help='Path to output directory')
    parser.add_argument('--save_video', action='store_true', help='Save output as video (output.avi)')
    parser.add_argument('--save_frames', action='store_true', help='Save output frames (frame_xxxx.jpg)')
    # --- Processing arguments ---
    parser.add_argument('--conf_thresh', type=float, default=0.4, help='Confidence threshold for detection')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--display', action='store_true', help='Display video/frames while processing')
    parser.add_argument('--classes', type=str, default='car,motorbike,person',
                        help='Comma-separated list of class names (must match training order)')

    args = parser.parse_args()

    # --- Validate input arguments ---
    # --- Validate input arguments ---
    print("--- Input Validation ---") # Added print
    input_provided = False
    video_path_exists = False
    if args.video:
        # --- Add these debug prints ---
        print(f"DEBUG: Received video path argument: '{args.video}'")
        try:
             path_exists = os.path.exists(args.video)
             print(f"DEBUG: os.path.exists() returned: {path_exists}")
             if path_exists:
                 video_path_exists = True # Use the checked value
                 input_provided = True
             else:
                  print(f"DEBUG: Path check failed for '{args.video}'") # Added print
        except Exception as e:
             print(f"DEBUG: Error during os.path.exists(): {e}") # Added print
        # --- End debug prints ---

    if args.image_dir and os.path.isdir(args.image_dir):
        if input_provided:
            print("Error: Please provide either --video OR --image_dir, not both.")
            sys.exit(1)
        input_provided = True
        print(f"DEBUG: Using image directory: '{args.image_dir}'") # Added print

    if args.camera_id >= 0:
        if input_provided:
             print("Error: Please provide only one input source (--video, --image_dir, or --camera_id).")
             sys.exit(1)
        input_provided = True
        print(f"DEBUG: Using camera ID: {args.camera_id}") # Added print


    if not input_provided:
        print("Error: No valid input source provided or path check failed.") # Modified error message
        print("Please specify --video <path>, --image_dir <path>, or --camera_id <id>.")
        sys.exit(1)
    # --- End Validation ---
    print("--- Input Validation OK ---") # Added print
    return args

def process_capture(cap, tracker, output_dir, save_video=True, save_frames=False, display=False, is_live=False):
    """Processes frames from a cv2.VideoCapture object (video file or camera)."""

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try: # Might fail on some streams
         fps = int(cap.get(cv2.CAP_PROP_FPS))
         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else -1
    except:
         fps = 30 # Default fps
         total_frames = -1 # Indicate unknown length

    print(f"Input Resolution: {width}x{height}")
    if fps > 0: print(f"Input FPS: {fps}")

    # Set up video writer if needed
    out = None
    output_video_path = None
    if save_video:
        output_video_path = os.path.join(output_dir, "tracked_output.avi")
        # Use 'XVID' for broader compatibility, especially on Windows
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 30, (width, height))
        if not out.isOpened():
            print(f"Error: Could not open video writer for {output_video_path}")
            print("Check if the codec 'XVID' is supported by your OpenCV installation.")
            save_video = False # Disable saving if writer fails
            out = None

    frame_count = 0
    while True: # Loop indefinitely for camera, breaks on video end or 'q'
        ret, frame = cap.read()
        if not ret:
            if not is_live: print("End of video file reached.")
            else: print("Error reading frame from camera.")
            break # Exit loop if no frame

        # --- Core Tracking Logic ---
        tracking_results = tracker.track(frame)
        result_frame = tracker.draw_tracks(frame.copy(), tracking_results)
        # --- End Tracking Logic ---

        # Write frame to output video
        if save_video and out is not None:
            out.write(result_frame)

        # Save frames if requested (e.g., save every 1 second)
        frame_save_interval = fps if fps > 0 else 30 # Approx 1 frame per second
        if save_frames and frame_count % frame_save_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, result_frame)

        # Display if requested
        if display:
            cv2.imshow('Object Tracking (ONNX + DeepSORT) - Press Q to Quit', result_frame)
            key = cv2.waitKey(1) & 0xFF # waitKey(1) crucial for video playback
            if key == ord('q'):
                print("Quit key pressed. Exiting...")
                break

        frame_count += 1

        # Display progress for videos
        if total_frames > 0 and frame_count % 30 == 0: # Update every 30 frames
            percent_complete = (frame_count / total_frames) * 100
            print(f"Processed frame {frame_count}/{total_frames} ({percent_complete:.1f}%)")
        elif is_live and frame_count % 100 == 0: # Update periodically for live feed
            print(f"Processed {frame_count} frames...")


    # Clean up
    cap.release()
    if out is not None:
        print(f"Releasing video writer for {output_video_path}...")
        out.release()
    if display:
        cv2.destroyAllWindows()

    if save_video and output_video_path:
        print(f"\nTracking completed. Output video saved to: {output_video_path}")
    elif save_frames:
         print(f"\nTracking completed. Output frames saved to: {output_dir}")
    else:
         print("\nTracking completed.")


def process_image_sequence(img_dir, tracker, output_dir, display=False):
    """Processes a sequence of images in a directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    try:
        all_files = os.listdir(img_dir)
    except FileNotFoundError:
        print(f"Error: Image directory not found: {img_dir}")
        return

    image_files = sorted([f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions])

    if not image_files:
        print(f"No images with supported extensions found in {img_dir}")
        return

    print(f"Found {len(image_files)} images to process.")

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Warning: Could not read image: {img_path}. Skipping.")
            continue

        # --- Core Tracking Logic ---
        tracking_results = tracker.track(frame)
        result_frame = tracker.draw_tracks(frame.copy(), tracking_results)
        # --- End Tracking Logic ---

        # Save the result frame
        output_path = os.path.join(output_dir, f"tracked_{img_file}")
        cv2.imwrite(output_path, result_frame)

         # Display if requested
        if display:
            cv2.imshow(f'Object Tracking - {img_file} (Press Q to Quit, Any other key for next)', result_frame)
            key = cv2.waitKey(0) & 0xFF # Wait indefinitely until a key is pressed
            if key == ord('q'):
                print("Quit key pressed. Exiting...")
                break # Exit the loop

        # Display progress
        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            percent_complete = ((i + 1) / len(image_files)) * 100
            print(f"Processed {i + 1}/{len(image_files)} images ({percent_complete:.1f}%)")

    if display:
        cv2.destroyAllWindows()
    print(f"\nImage sequence processing completed. Results saved to: {output_dir}")


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {os.path.abspath(args.output_dir)}")

    # Parse class names
    class_names = [name.strip() for name in args.classes.split(',') if name.strip()]
    print(f"Tracking classes: {class_names}")
    if not class_names:
         print("Error: No valid class names provided via --classes argument.")
         sys.exit(1)


    # --- Initialize ONNX tracker ---
    print("Initializing Object Tracker...")
    try:
        tracker = ObjectTracker(
            onnx_model_path=args.model, # Pass ONNX model path
            class_names=class_names,
            confidence_threshold=args.conf_thresh,
            iou_threshold=args.iou_thresh
        )
        print("Tracker initialized successfully.")
    except Exception as e:
         print(f"Fatal error initializing tracker: {e}")
         sys.exit(1)
    # --- End Tracker Initialization ---

    # --- Process Input ---
    if args.video:
        print(f"Processing video: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.video}")
        else:
            process_capture(cap, tracker, args.output_dir, args.save_video, args.save_frames, args.display, is_live=False)

    elif args.image_dir:
        print(f"Processing images in directory: {args.image_dir}")
        process_image_sequence(args.image_dir, tracker, args.output_dir, args.display)

    elif args.camera_id >= 0:
        print(f"Processing live feed from camera ID: {args.camera_id}")
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
             print(f"Error: Could not open camera {args.camera_id}. Check if it's connected and drivers are installed.")
        else:
            # Typically don't save video from live feed by default unless requested
            save_live_video = args.save_video # Respect command line flag
            process_capture(cap, tracker, args.output_dir, save_live_video, args.save_frames, display=True, is_live=True) # Force display for live

    else:
        # This case should be caught by arg parsing validation, but as a fallback:
        print("Error: No valid input source specified in arguments.")

if __name__ == "__main__":
    main()