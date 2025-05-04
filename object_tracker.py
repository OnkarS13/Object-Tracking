# object_tracker.py

import os
import cv2
import numpy as np
import torch # Still needed for DeepSORT feature extractor if using GPU
import onnxruntime as ort # Import ONNX Runtime
from collections import defaultdict
import sys
import urllib.request
import zipfile # For extracting downloaded repos

# --- DeepSORT Setup (Downloads/Checks - Modified for better error handling) ---
def download_and_extract(repo_url, repo_name, zip_name):
    """Downloads and extracts a GitHub repository."""
    if not os.path.exists(repo_name):
        print(f"Downloading {repo_name} repository...")
        zip_url = f"{repo_url}/archive/refs/heads/master.zip"
        zip_path = f"{zip_name}.zip"
        extracted_folder_name = f"{zip_name}-master" # Default GitHub zip structure

        try:
            urllib.request.urlretrieve(zip_url, zip_path)
            print(f"Downloaded {zip_path}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            print(f"Extracted to {extracted_folder_name}")

            # Rename folder
            if os.path.exists(extracted_folder_name):
                 os.rename(extracted_folder_name, repo_name)
                 print(f"Renamed {extracted_folder_name} to {repo_name}")
            else:
                print(f"Error: Extracted folder {extracted_folder_name} not found.")
                raise FileNotFoundError

            # Clean up zip file
            os.remove(zip_path)
            print(f"Removed {zip_path}")
            print(f"{repo_name} setup complete.")

        except Exception as e:
            print(f"Error setting up {repo_name}: {e}")
            print(f"Please manually clone: {repo_url}.git into a folder named '{repo_name}'")
            # Clean up potentially incomplete download/extraction
            if os.path.exists(zip_path): os.remove(zip_path)
            if os.path.exists(extracted_folder_name): os.rmdir(extracted_folder_name) # Use rmdir if empty, or shutil.rmtree
            sys.exit(1) # Exit if setup fails

# Download DeepSORT Core
download_and_extract("https://github.com/nwojke/deep_sort", "deep_sort", "deep_sort")
# Download DeepSORT PyTorch (for feature extractor)
download_and_extract("https://github.com/ZQPei/deep_sort_pytorch", "deep_sort_pytorch", "deep_sort_pytorch")

# Add to Python path
if os.path.exists('deep_sort'): sys.path.append(os.path.abspath('deep_sort'))
if os.path.exists('deep_sort_pytorch'): sys.path.append(os.path.abspath('deep_sort_pytorch'))

# --- Import DeepSORT components ---
try:
    from deep_sort import nn_matching
    # Adjust import path based on actual structure in nwojke/deep_sort
    from deep_sort_pytorch.deep_sort.sort.detection import Detection
    from deep_sort_pytorch.deep_sort.sort.tracker import Tracker
    # Adjust import path based on actual structure in ZQPei/deep_sort_pytorch
    from deep_sort_pytorch.deep_sort.deep.feature_extractor import Extractor
except ImportError as e:
    print(f"ImportError: {e}")
    print("Could not import DeepSORT components. Ensure repositories exist and paths are correct.")
    print("Expected structure: ./deep_sort/ and ./deep_sort_pytorch/")
    sys.exit(1)

# --- Helper Function: Preprocessing for ONNX YOLOv8 ---
def preprocess_image_onnx(img, input_width, input_height):
    """
    Preprocesses an image for YOLOv8 ONNX inference.
    Handles reading, resizing with letterboxing, BGR->RGB, normalization, and CHW conversion.
    Returns the processed image tensor, original image shape, and scaling/padding info.
    """
    original_image = img # Input is already a cv2 image (frame)
    original_height, original_width = original_image.shape[:2]

    # Calculate scaling factor and new size
    ratio = min(input_width / original_width, input_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize image
    resized_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    pad_width = input_width - new_width
    pad_height = input_height - new_height
    top_pad, bottom_pad = pad_height // 2, pad_height - (pad_height // 2)
    left_pad, right_pad = pad_width // 2, pad_width - (pad_width // 2)

    # Apply letterboxing (padding)
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, value=(114, 114, 114)) # Gray padding

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

    # Normalize (0-255 -> 0.0-1.0)
    normalized_image = rgb_image.astype(np.float32) / 255.0

    # Transpose (HWC -> CHW)
    chw_image = np.transpose(normalized_image, (2, 0, 1))

    # Add batch dimension (CHW -> NCHW)
    input_tensor = np.expand_dims(chw_image, axis=0)

    return input_tensor, (original_width, original_height), (ratio, left_pad, top_pad)

# --- Helper Function: Postprocessing for ONNX YOLOv8 ---
def postprocess_output_onnx(output_data, conf_threshold, iou_threshold, original_shape, scale_pad_info, class_names):
    """
    Postprocesses the raw output tensor from YOLOv8 ONNX model.
    Handles reshaping, confidence filtering, NMS, and coordinate scaling.
    IMPORTANT: Returns detections in [x, y, w, h] format for DeepSORT.
    """
    # Output shape is typically [batch_size, num_classes + 4, num_proposals]
    # Example: [1, 3+4, 8400] for nc=3 and 640x640 input
    if not output_data.any(): # Handle empty output
        return []
        
    output_data = output_data[0].T  # Transpose to [num_proposals, num_classes + 4]

    boxes = []
    scores = []
    class_ids = []

    # Extract coordinates, confidence, and class probabilities
    box_coords = output_data[:, :4] # cx, cy, w, h
    all_class_scores = output_data[:, 4:]

    original_width, original_height = original_shape
    ratio, left_pad, top_pad = scale_pad_info

    # Iterate through proposals
    for i in range(output_data.shape[0]):
        class_scores = all_class_scores[i]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence >= conf_threshold:
            cx, cy, w, h = box_coords[i]

            # Convert center coords (relative to padded/resized image) to x1, y1, x2, y2
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            boxes.append([x1, y1, x2, y2]) # Store xyxy for NMS input scaling later
            scores.append(float(confidence))
            class_ids.append(int(class_id))

    if not boxes:
        return []

    # Perform Non-Maximum Suppression (NMS) using xyxy format relative to padded/resized input
    # Convert boxes to format required by cv2.dnn.NMSBoxes: [x_center, y_center, width, height] is NOT correct here.
    # It expects [x_min, y_min, width, height]. Let's use the xyxy boxes directly and adapt.
    nms_boxes_xyxy = np.array(boxes).astype(int) # Use integer coords for NMS function if needed
    
    # NMSBoxes requires boxes in (x, y, w, h) format, NOT xyxy
    # Calculate w, h from xyxy for NMS input
    nms_boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in nms_boxes_xyxy]


    indices = cv2.dnn.NMSBoxes(nms_boxes_xywh, scores, conf_threshold, iou_threshold)

    detections = []
    if len(indices) > 0:
        # Handle cases where indices might be a nested list/tuple
        if isinstance(indices[0], (list, tuple)):
            indices = [item[0] for item in indices] # Flatten if necessary

        for i in indices:
            # Get the original xyxy box corresponding to the kept index
            x1, y1, x2, y2 = boxes[i]
            class_id = class_ids[i]
            confidence = scores[i]

            # Scale coordinates back to original image size, accounting for padding
            orig_x1 = (x1 - left_pad) / ratio
            orig_y1 = (y1 - top_pad) / ratio
            orig_x2 = (x2 - left_pad) / ratio
            orig_y2 = (y2 - top_pad) / ratio

            # Clip coordinates to image bounds
            orig_x1 = max(0, min(orig_x1, original_width))
            orig_y1 = max(0, min(orig_y1, original_height))
            orig_x2 = max(0, min(orig_x2, original_width))
            orig_y2 = max(0, min(orig_y2, original_height))

            # Calculate final width and height for DeepSORT format (xywh)
            final_w = orig_x2 - orig_x1
            final_h = orig_y2 - orig_y1

            # Append detection in the required [x, y, w, h] format
            if class_id < len(class_names): # Ensure class_id is valid
                detections.append({
                    'box': [int(orig_x1), int(orig_y1), int(final_w), int(final_h)], # x, y, w, h
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_names[class_id]
                })

    return detections


# --- Object Tracker Class ---
class ObjectTracker:
    def __init__(self, onnx_model_path, class_names, confidence_threshold=0.5, iou_threshold=0.45):
        self.class_names = class_names
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_width = 640  # Assuming model trained with 640x640
        self.input_height = 640

        # --- Load ONNX Model ---
        print(f"Loading ONNX model from {onnx_model_path}")
        try:
            # Check available providers and prioritize GPU if available
            print(f"Available ONNX Runtime providers: {ort.get_available_providers()}")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            print(f"Attempting to use providers: {providers}")
            self.session = ort.InferenceSession(onnx_model_path, providers=providers)
            print(f"ONNX model loaded successfully using {self.session.get_providers()}.")

            # Get model input details
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            model_input_shape = self.session.get_inputs()[0].shape # e.g., [1, 3, 640, 640]
            # You might want to verify model_input_shape matches self.input_width/height
            print(f"Model Input: name={self.input_name}, shape={model_input_shape}")
            print(f"Model Output: name={self.output_name}, shape={self.session.get_outputs()[0].shape}")

        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            print("Ensure ONNX Runtime is installed (pip install onnxruntime or onnxruntime-gpu).")
            print("Also verify the model path and file integrity.")
            sys.exit(1)
        # --- End ONNX Model Loading ---

        # Initialize colors for visualization
        np.random.seed(42) # for consistent colors
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

        # Initialize DeepSORT
        max_cosine_distance = 0.4
        nn_budget = 100
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

        # --- Feature Extractor Setup ---
        # Use absolute paths for reliability
        extractor_dir = os.path.join(os.path.abspath("deep_sort_pytorch"), "deep_sort", "deep", "checkpoint")
        extractor_model_path = os.path.join(extractor_dir, "ckpt.t7")

        if not os.path.exists(extractor_model_path):
            os.makedirs(extractor_dir, exist_ok=True)
            print(f"Downloading feature extractor model to {extractor_model_path}...")
            # Using the known working Google Drive link
            model_url = "https://drive.google.com/uc?id=1_qIVizmFnXXCHVwF9KCpvlWTQYhvNiVK&export=download"
            try:
                # Using requests for potentially better handling of redirects/large files
                import requests
                response = requests.get(model_url, stream=True)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                with open(extractor_model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download completed!")
            except Exception as e:
                print(f"Error downloading feature extractor model: {e}")
                print("Please manually download the model file from:")
                print(model_url)
                print(f"And save it to: {extractor_model_path}")
                if os.path.exists(extractor_model_path): os.remove(extractor_model_path) # Clean up failed download
                sys.exit(1)

        # Initialize the feature extractor
        use_cuda_extractor = torch.cuda.is_available()
        print(f"Initializing DeepSORT feature extractor (use_cuda={use_cuda_extractor})...")
        try:
             self.extractor = Extractor(extractor_model_path, use_cuda=use_cuda_extractor)
             print("Feature extractor initialized.")
        except Exception as e:
             print(f"Error initializing feature extractor: {e}")
             print("Please check the ckpt.t7 file and dependencies.")
             sys.exit(1)
        # --- End Feature Extractor Setup ---

        # For counting objects
        self.object_counter = defaultdict(int)
        self.tracked_objects = {} # Stores {track_id: class_name} for counting


    def track(self, frame):
        # 1. Preprocess frame for ONNX model
        input_tensor, original_shape, scale_pad_info = preprocess_image_onnx(
            frame, self.input_width, self.input_height
        )

        # 2. Run ONNX Inference
        try:
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            raw_detections = outputs[0]
        except Exception as e:
             print(f"ONNX Runtime inference error: {e}")
             return [] # Return empty if inference fails

        # 3. Postprocess ONNX output to get detections
        detections = postprocess_output_onnx(
            raw_detections,
            self.conf_threshold,
            self.iou_threshold,
            original_shape, # Pass original (width, height)
            scale_pad_info,
            self.class_names # Pass class names for the function
        )

        # 4. Prepare data for DeepSORT and Extract Features
        # --- Feature Extraction Modification ---
        if len(detections) > 0:
            # Get boxes in [x, y, w, h] format from detections
            xywh_boxes = np.array([d['box'] for d in detections])
            scores = np.array([d['confidence'] for d in detections])
            classes = np.array([d['class_id'] for d in detections])

            # Create crops from the original frame based on xywh boxes
            im_crops = []
            original_h, original_w = frame.shape[:2] # Get frame dimensions

            for box in xywh_boxes:
                x, y, w, h = map(int, box)
                # Clamp coordinates to frame dimensions before cropping
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(original_w, x + w)
                y2 = min(original_h, y + h)
                # Check if the box has valid dimensions after clamping
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    im_crops.append(crop)
                else:
                    # Handle invalid box (e.g., log a warning or append a placeholder if necessary)
                    # Appending a placeholder might require extractor to handle it.
                    # Safest might be to skip feature extraction for this box,
                    # but that complicates matching features back to detections.
                    # For now, let's append if valid. If im_crops is empty later, we handle it.
                    print(f"Warning: Skipping invalid box for feature extraction: {[x1,y1,x2,y2]}")


            # Check if we have any valid crops to extract features from
            if not im_crops:
                    print("No valid crops found for feature extraction in this frame.")
                    # If no crops, skip tracker update for this frame? Or predict only?
                    # For now, let's just proceed; tracker update will handle empty detections.
                    features = [] # No features extracted
                    # Make sure detection_list remains empty or is handled correctly
                    detection_list = [] # No detections with features to update tracker
            else:
                # Extract features from the *list of crops*
                try:
                    features = self.extractor(im_crops) # Pass only the list of crops
                except Exception as e:
                    print(f"Feature extraction error: {e}")
                    # Handle error, e.g., skip tracking for this frame
                    return [] # Return empty tracking results for this frame

            # Check if number of features matches number of valid crops (should match detections)
            # This assumes extractor returns one feature vector per crop.
            # Need to realign features with the original detections if some crops were skipped.
            # For simplicity now, assume features correspond to the original detections list length
            # IF NO CROPS WERE SKIPPED. A more robust solution is needed if skipping happens.
            if len(features) != len(detections) and len(im_crops) == len(detections):
                    print(f"Warning: Mismatch between number of detections ({len(detections)}) and extracted features ({len(features)}). Skipping frame.")
                    return []


            # Prepare DeepSORT Detection objects (requires xywh)
            detection_list = []
            # Ensure we only create detections for which features were successfully extracted
            # Inside the loop where you iterate through features
            # Inside the loop where you iterate through features
            for i in range(len(features)):
                bbox_xywh = xywh_boxes[i]
                score = scores[i]
                cls_id = classes[i] # The integer class ID (e.g., 0, 1, 2)
                feature = features[i]
                # class_name = self.class_names[cls_id] # We don't need the name for the constructor itself

                # Instantiate the ZQPei Detection class with the CLASS ID
                # print(f"Creating Detection: box={bbox_xywh}, score={score:.2f}, id={cls_id}") # Debug print updated
                detection_list.append(Detection(bbox_xywh, score, cls_id, feature)) # Pass integer cls_id
            # --- End Feature Extraction Modification ---


            # 5. Update DeepSORT Tracker (This part remains the same)
            if detection_list: # Only update if we have valid detections with features
                    self.tracker.predict()
                    self.tracker.update(detection_list)
            else:
                    # If no valid detections, still call predict to advance Kalman filters
                    self.tracker.predict()
                    print("No valid detections to update tracker.")


            # 6. Process Tracking Results (This part remains mostly the same)
            # ... (rest of the track method follows) ...

            # 6. Process Tracking Results
            tracking_results = []
            current_frame_track_ids = set() # Keep track of active IDs in this frame

            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = track.track_id
                current_frame_track_ids.add(track_id) # Mark as active

                # Use the class ID stored in the track
                # The detection that matched this track might have had a slightly different class
                # if your detector is noisy, but DeepSORT keeps the initial class.
                class_id = track.cls # Access the class ID directly from the attribute # Use DeepSORT's stored class ID
                if not isinstance(class_id, int): # Add safety check for type
                    print(f"Warning: Track {track_id} has non-integer class_id '{class_id}' ({type(class_id)}). Skipping.")
                    continue
                if class_id >= len(self.class_names): # Safety check for range
                    print(f"Warning: Track {track_id} has invalid class_id {class_id}. Skipping.")
                    continue

                bbox = track.to_tlwh()  # Get current estimated box (x, y, w, h)
                confidence = track.confidence if hasattr(track, 'confidence') else scores[np.argmin(np.sum(np.abs(xywh_boxes - bbox), axis=1))] # Estimate confidence if not available

                # Add to results
                tracking_results.append({
                    'track_id': track_id,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id],
                    'bbox': bbox, # xywh
                    'confidence': confidence
                })

                # Count objects only once when they first appear and are confirmed
                obj_name = self.class_names[class_id]
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = obj_name
                    self.object_counter[obj_name] += 1
                    print(f"New object tracked: {obj_name} (ID: {track_id}). Total {obj_name}s: {self.object_counter[obj_name]}")

            # Optional: Clean up tracked_objects dictionary for IDs that are no longer tracked
            # (Could be done less frequently)
            # lost_ids = set(self.tracked_objects.keys()) - current_frame_track_ids
            # for lost_id in lost_ids:
            #     if lost_id in self.tracked_objects: # Check existence before deleting
            #         print(f"Track ID {lost_id} lost.")
                     # Maybe add logic here if an object re-appears later
            #         # For simple counting, we don't decrement here.
            #         pass # Keep the count, just note it's lost for now

            return tracking_results

        # Return empty list if no detections were made initially
        return []


    def draw_tracks(self, frame, tracking_results):
        height, width = frame.shape[:2] # Get frame dimensions for text placement

        for result in tracking_results:
            track_id = result['track_id']
            class_id = result['class_id']
            class_name = result['class_name']
            bbox = result['bbox'] # xywh

            x, y, w, h = [int(v) for v in bbox]

            # Ensure box coordinates are valid
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)

            # Check if the box has valid dimensions after clamping
            if x2 <= x1 or y2 <= y1:
                 continue

            # Get color for this class
            color = [int(c) for c in self.colors[class_id % len(self.colors)]] # Use modulo for safety

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label (Class-ID)
            label = f"{class_name}-{track_id}"
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Ensure label background doesn't go out of bounds
            label_x1 = x1
            label_y1 = max(text_height + 4, y1) # Position above the box, but ensure y1 is valid
            label_x2 = label_x1 + text_width
            label_y2 = label_y1 - text_height - 4

            # Adjust if label goes offscreen vertically
            if label_y2 < 0:
                label_y1 = y1 + text_height + 4 # Place below box if no space above
                label_y2 = y1

            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1) # Filled background
            cv2.putText(frame, label, (label_x1, label_y1 - baseline + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness) # White text

        # Draw object counts (persistent total counts)
        y_offset = 30 # Start text lower down
        for i, (obj_name, count) in enumerate(sorted(self.object_counter.items())): # Sort for consistent order
            text = f"Total {obj_name}: {count}"
            cv2.putText(frame, text, (15, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Slightly larger font

        return frame