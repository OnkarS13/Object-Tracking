import sys
import os
import cv2
import numpy as np
import time # For optional sleep
from concurrent.futures import ThreadPoolExecutor, Future
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QRadioButton,
    QGroupBox, QSlider, QCheckBox, QSizePolicy, QSpacerItem, QMessageBox,
    QButtonGroup
)
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSize, QMetaObject, Q_ARG # Import QMetaObject, Q_ARG

from PIL import Image # For placeholder

# --- Worker Class (No QObject/QThread) ---

class TrackingWorker:
    """
    Worker class that performs tracking. Designed to be run in a ThreadPoolExecutor.
    Uses QMetaObject.invokeMethod to communicate back to the main GUI thread.
    """
    def __init__(self, main_window, tracker_instance, input_path, camera_id, output_dir, save_frames, is_live, process_images):
        # Store reference to main window to invoke its slots
        self.main_window = main_window
        self.tracker = tracker_instance
        self.input_path = input_path
        self.camera_id = camera_id
        self.output_dir = output_dir
        self.save_frames = save_frames
        self.is_live = is_live
        self.process_images = process_images
        self._is_running = True

    def run(self):
        """Main execution logic. Calls appropriate processing function."""
        try:
            if self.process_images:
                self.run_image_sequence_processing()
            else:
                self.run_video_processing()
        except Exception as e:
            # Report error back to main thread if run fails catastrophically
            error_msg = f"Critical worker error: {e}"
            print(error_msg)
            QMetaObject.invokeMethod(
                self.main_window, "_handle_error", Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, error_msg)
            )
        finally:
            # Note: Finish signal is handled by the Future's done callback
            print("Worker run method finished.")


    def stop(self):
        """Signals the worker to stop processing."""
        print("Worker stop requested.")
        self._is_running = False
        # Use invokeMethod to update status safely from whichever thread calls stop
        QMetaObject.invokeMethod(
            self.main_window, "_handle_progress", Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, "Stopping...")
        )


    # --- Safe GUI Update Methods ---
    def _emit_progress(self, message):
        QMetaObject.invokeMethod(
            self.main_window, "_handle_progress", Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, message)
        )

    def _emit_frame(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0: return
        try:
            # Convert BGR to RGB first
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            # Use .copy() - Crucial! Prevents QImage pointing to data that might change/be freed
            # Ensure data is contiguous C-style array for QImage
            if not frame_rgb.flags['C_CONTIGUOUS']:
                frame_rgb = np.ascontiguousarray(frame_rgb)
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

            if qt_image.isNull():
                 print("Warning: Created QImage is null.")
                 return

            QMetaObject.invokeMethod(
                self.main_window, "_handle_frame_qimage", Qt.ConnectionType.QueuedConnection,
                Q_ARG(QImage, qt_image) # Pass QImage directly
            )
        except Exception as e:
            print(f"Error converting/emitting frame: {e}")


    def _emit_error(self, message):
         QMetaObject.invokeMethod(
            self.main_window, "_handle_error", Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, message)
         )

    # --- Processing Loops ---
    def run_video_processing(self):
        """Processes video frames from file or camera."""
        cap = None
        try:
            if self.is_live:
                self._emit_progress(f"Opening camera: {self.camera_id}")
                cap = cv2.VideoCapture(self.camera_id)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            else:
                self._emit_progress(f"Opening video file: {self.input_path}")
                cap = cv2.VideoCapture(self.input_path)

            if not cap or not cap.isOpened():
                raise IOError(f"Cannot open video source: {self.input_path or self.camera_id}")

            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS) if not self.is_live else 30
            save_interval = int(fps) if fps > 0 else 30

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    self._emit_progress("End of video or cannot read frame.")
                    break
                if frame is None or frame.size == 0: continue

                try:
                    tracking_results = self.tracker.track(frame)
                    result_frame = self.tracker.draw_tracks(frame.copy(), tracking_results)
                except Exception as track_e:
                    print(f"Error during tracker processing: {track_e}")
                    self._emit_error(f"Tracking error: {track_e}")
                    result_frame = frame # Show original on error

                self._emit_frame(result_frame) # Emit the processed BGR frame

                if self.save_frames and frame_count % save_interval == 0:
                    try:
                        frame_path = os.path.join(self.output_dir, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(frame_path, result_frame)
                    except Exception as e: print(f"Error saving frame {frame_count}: {e}")

                frame_count += 1
                # time.sleep(0.001) # Small sleep if needed, avoid blocking GUI thread

        except Exception as e:
            print(f"Error in video processing loop: {e}")
            self._emit_error(f"Video processing error: {e}")
        finally:
            if cap: cap.release()
            print("Video processing in worker finished.")


    def run_image_sequence_processing(self):
        """Processes a sequence of images."""
        try:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            try: all_files = os.listdir(self.input_path)
            except FileNotFoundError: raise FileNotFoundError(f"Image directory not found: {self.input_path}")

            image_files = sorted([f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions])
            if not image_files: raise FileNotFoundError(f"No supported images found in {self.input_path}")

            num_images = len(image_files)
            self._emit_progress(f"Found {num_images} images.")

            for i, img_file in enumerate(image_files):
                if not self._is_running:
                    self._emit_progress("Image processing stopped.")
                    break

                self._emit_progress(f"Processing {i+1}/{num_images}: {img_file}")
                img_path = os.path.join(self.input_path, img_file)
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Warning: Could not read image: {img_path}. Skipping.")
                    continue

                try:
                     tracking_results = self.tracker.track(frame)
                     result_frame = self.tracker.draw_tracks(frame.copy(), tracking_results)
                except Exception as track_e:
                     print(f"Error during tracker processing for {img_file}: {track_e}")
                     self._emit_error(f"Tracking error on {img_file}: {track_e}")
                     result_frame = frame # Show original on error

                self._emit_frame(result_frame) # Emit BGR frame

                if self.save_frames:
                    try:
                        output_path = os.path.join(self.output_dir, f"tracked_{img_file}")
                        cv2.imwrite(output_path, result_frame)
                    except Exception as e: print(f"Error saving tracked image {img_file}: {e}")

                time.sleep(0.05) # Add delay for viewing image sequence

        except Exception as e:
            print(f"Error in image processing loop: {e}")
            self._emit_error(f"Image processing error: {e}")
        finally:
            print("Image processing in worker finished.")


# --- Main Application Window ---

class TrackingAppGUI(QMainWindow):
    """Main GUI Window using PySide6 and ThreadPoolExecutor."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Tracking GUI (PySide6 - Executor)")
        self.setGeometry(100, 100, 950, 800)

        # --- State ---
        self.tracker_instance = None
        self.worker_instance = None # Reference to the worker instance
        self.current_future = None # Reference to the Future object
        self.is_tracking = False
        # Use a ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=1) # Only 1 worker for sequential processing

        # --- Placeholder Image ---
        self.placeholder_img = QImage(640, 480, QImage.Format.Format_RGB888)
        self.placeholder_img.fill(Qt.GlobalColor.lightGray)
        self.placeholder_pixmap = QPixmap.fromImage(self.placeholder_img)

        # --- Initialize ObjectTracker Class Reference ---
        self.ObjectTracker = None
        try:
            # Ensure object_tracker.py is importable
            from object_tracker import ObjectTracker as OT
            self.ObjectTracker = OT
        except ImportError as e:
            # Display error clearly if import fails
            err_msg = f"Could not import ObjectTracker: {e}\n\n" \
                      f"Please ensure:\n" \
                      f"1. 'object_tracker.py' is in the same directory as this GUI script or in the Python path.\n" \
                      f"2. All dependencies for 'object_tracker.py' (like deep_sort folders, PyTorch, ONNX Runtime, etc.) are correctly installed and accessible.\n" \
                      f"3. The 'deep_sort' and 'deep_sort_pytorch' folders have the correct structure."
            QMessageBox.critical(self, "Import Error", err_msg)
            # Exit gracefully if core component is missing
            # Use QTimer to allow the message box to show before exiting
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, sys.exit) # Exit after 100ms
            # sys.exit(1) # Direct exit might close the message box too quickly

        except Exception as e:
             QMessageBox.critical(self, "Initialization Error", f"An unexpected error occurred during initial imports: {e}")
             from PySide6.QtCore import QTimer
             QTimer.singleShot(100, sys.exit)
             # sys.exit(1)

        # --- Setup UI ---
        self.init_ui()
        self.apply_stylesheet()

    def init_ui(self):
        """Create and arrange UI elements."""
        # --- Identical to the previous QThread version ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Configuration Group
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        # Row 1: Model Path
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("ONNX Model:"))
        self.model_entry = QLineEdit()
        self.model_entry.setPlaceholderText("Path to .onnx file")
        model_layout.addWidget(self.model_entry)
        model_browse_btn = QPushButton("Browse...")
        model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(model_browse_btn)
        config_layout.addLayout(model_layout)
        # Row 2: Input Source
        input_group = QGroupBox("Input Source")
        input_layout = QHBoxLayout()
        self.input_button_group = QButtonGroup(self)
        self.radio_camera = QRadioButton("Camera ID:")
        self.radio_camera.setChecked(True)
        self.cam_id_entry = QLineEdit("0")
        self.cam_id_entry.setFixedWidth(50)
        self.radio_video = QRadioButton("Video File:")
        self.video_entry = QLineEdit()
        self.video_entry.setPlaceholderText("Path to video file")
        self.video_entry.setEnabled(False)
        self.video_browse_btn = QPushButton("Browse...")
        self.video_browse_btn.setEnabled(False)
        self.video_browse_btn.clicked.connect(self.browse_video)
        self.radio_image_dir = QRadioButton("Image Dir:")
        self.img_dir_entry = QLineEdit()
        self.img_dir_entry.setPlaceholderText("Path to image directory")
        self.img_dir_entry.setEnabled(False)
        self.img_dir_browse_btn = QPushButton("Browse...")
        self.img_dir_browse_btn.setEnabled(False)
        self.img_dir_browse_btn.clicked.connect(self.browse_image_dir)
        self.input_button_group.addButton(self.radio_camera)
        self.input_button_group.addButton(self.radio_video)
        self.input_button_group.addButton(self.radio_image_dir)
        self.input_button_group.buttonClicked.connect(self.update_input_state)
        input_layout.addWidget(self.radio_camera)
        input_layout.addWidget(self.cam_id_entry)
        input_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Policy.Expanding))
        input_layout.addWidget(self.radio_video)
        input_layout.addWidget(self.video_entry)
        input_layout.addWidget(self.video_browse_btn)
        input_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Policy.Expanding))
        input_layout.addWidget(self.radio_image_dir)
        input_layout.addWidget(self.img_dir_entry)
        input_layout.addWidget(self.img_dir_browse_btn)
        input_group.setLayout(input_layout)
        config_layout.addWidget(input_group)
        # Row 3: Output Directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Dir:"))
        self.output_entry = QLineEdit("./tracking_output_gui")
        output_layout.addWidget(self.output_entry)
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(output_browse_btn)
        config_layout.addLayout(output_layout)
        # Row 4: Parameters
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Conf:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100); self.conf_slider.setValue(40)
        self.conf_slider.setFixedWidth(100)
        self.conf_label = QLabel(f"{self.conf_slider.value()/100.0:.2f}")
        self.conf_slider.valueChanged.connect(lambda val: self.conf_label.setText(f"{val/100.0:.2f}"))
        param_layout.addWidget(self.conf_slider); param_layout.addWidget(self.conf_label)
        param_layout.addSpacerItem(QSpacerItem(20, 10))
        param_layout.addWidget(QLabel("IoU:"))
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(0, 100); self.iou_slider.setValue(45)
        self.iou_slider.setFixedWidth(100)
        self.iou_label = QLabel(f"{self.iou_slider.value()/100.0:.2f}")
        self.iou_slider.valueChanged.connect(lambda val: self.iou_label.setText(f"{val/100.0:.2f}"))
        param_layout.addWidget(self.iou_slider); param_layout.addWidget(self.iou_label)
        param_layout.addSpacerItem(QSpacerItem(20, 10))
        param_layout.addWidget(QLabel("Classes:"))
        self.classes_entry = QLineEdit("car,motorbike,person")
        param_layout.addWidget(self.classes_entry)
        param_layout.addSpacerItem(QSpacerItem(20, 10))
        self.save_frames_check = QCheckBox("Save Frames")
        param_layout.addWidget(self.save_frames_check)
        param_layout.addStretch(1)
        config_layout.addLayout(param_layout)
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # Display Group
        display_group = QGroupBox("Live Feed / Output")
        display_layout = QVBoxLayout()
        self.video_display_label = QLabel("Video feed will appear here.")
        self.video_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display_label.setMinimumSize(640, 480)
        self.video_display_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_display_label.setPixmap(self.placeholder_pixmap)
        self.video_display_label.setStyleSheet("background-color: #333; border: 1px solid #555;")
        display_layout.addWidget(self.video_display_label)
        display_group.setLayout(display_layout)
        main_layout.addWidget(display_group, stretch=1)

        # Control Group
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Tracking")
        self.start_button.clicked.connect(self.start_tracking)
        self.stop_button = QPushButton("Stop Tracking")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_tracking)
        self.status_label = QLabel("Status: Idle")
        control_layout.addWidget(self.start_button); control_layout.addWidget(self.stop_button)
        control_layout.addStretch(1); control_layout.addWidget(self.status_label)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        self.update_input_state() # Initial setup

    def apply_stylesheet(self):
        """Applies a basic stylesheet for a cleaner look."""
        # --- Added explicit QLabel color ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5; /* Light grey background */
            }
            QGroupBox {
                font-weight: bold;
                color: #333; /* Darker text for group box title */
                background-color: #ffffff; /* White background for group boxes */
                border: 1px solid #e0e0e0; /* Lighter border */
                border-radius: 5px;
                margin-top: 10px;
                padding: 20px 10px 10px 10px; /* Adjusted padding (more top for title) */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px; /* Position title */
                background-color: #ffffff; /* Ensure title background matches group box */
                color: #111; /* Explicit dark color for title */
            }
            QLabel {
                font-size: 10pt; /* Consistent font size */
                color: #212121; /* Explicit dark grey color for all labels */
                background-color: transparent; /* Ensure label background is transparent */
                padding: 2px; /* Add slight padding */
            }
            /* Specific styling for labels within the display group if needed */
            #video_display_label {
                 color: white; /* Example if text was shown on dark background */
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #bdbdbd; /* Grey border */
                border-radius: 3px;
                font-size: 10pt;
                background-color: #ffffff; /* White background for entries */
                color: #212121; /* Dark text */
            }
            QLineEdit:focus {
                border: 1px solid #42a5f5; /* Blue border on focus */
            }
            QLineEdit:disabled {
                background-color: #eeeeee; /* Lighter grey when disabled */
                color: #9e9e9e;
            }
            QPushButton {
                background-color: #42a5f5; /* Blue background */
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 3px;
                font-size: 10pt;
                font-weight: bold;
                min-width: 80px; /* Minimum width */
            }
            QPushButton:hover {
                background-color: #1e88e5; /* Darker blue on hover */
            }
            QPushButton:pressed {
                background-color: #1565c0; /* Even darker blue when pressed */
            }
            QPushButton:disabled {
                background-color: #e0e0e0; /* Grey when disabled */
                color: #bdbdbd; /* Lighter text when disabled */
            }
            QRadioButton, QCheckBox {
                font-size: 10pt;
                color: #212121; /* Dark text for radio/check */
                spacing: 5px; /* Space between button and text */
            }
            QSlider::groove:horizontal {
                border: 1px solid #bdbdbd;
                height: 5px; /* Groove height */
                background: #e0e0e0;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #42a5f5; /* Blue handle */
                border: 1px solid #42a5f5;
                width: 14px; /* Handle width */
                margin: -5px 0; /* Center handle vertically */
                border-radius: 7px; /* Circular handle */
            }
            #status_label { /* Use object name for specific styling */
                font-style: italic;
                color: #616161; /* Dark grey */
                font-size: 9pt; /* Slightly smaller status */
            }
        """)
        # Set object names for specific styling if needed
        self.status_label.setObjectName("status_label")
        self.video_display_label.setObjectName("video_display_label")


    # --- Browse Slots (Identical to previous version) ---
    @Slot()
    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX Model", "", "ONNX Files (*.onnx)")
        if path: self.model_entry.setText(path)
    @Slot()
    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if path: self.video_entry.setText(path)
    @Slot()
    def browse_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if path: self.img_dir_entry.setText(path)
    @Slot()
    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path: self.output_entry.setText(path)

    # --- State Update Slot (Identical to previous version) ---
    @Slot()
    def update_input_state(self):
        self.cam_id_entry.setEnabled(self.radio_camera.isChecked())
        self.video_entry.setEnabled(self.radio_video.isChecked())
        self.video_browse_btn.setEnabled(self.radio_video.isChecked())
        self.img_dir_entry.setEnabled(self.radio_image_dir.isChecked())
        self.img_dir_browse_btn.setEnabled(self.radio_image_dir.isChecked())

    # --- Tracking Control Slots ---
    @Slot()
    def start_tracking(self):
        """Validates inputs and submits the tracking worker to the ThreadPoolExecutor."""
        if self.is_tracking:
            QMessageBox.warning(self, "Warning", "Tracking is already in progress.")
            return
        if self.ObjectTracker is None: # Check if class was loaded
             QMessageBox.critical(self, "Error", "ObjectTracker class not loaded. Cannot start.")
             return

        # --- Get Values and Validate (Identical to previous version) ---
        model_path = self.model_entry.text(); output_dir = self.output_entry.text()
        classes_str = self.classes_entry.text(); conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0; save_frames = self.save_frames_check.isChecked()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", "Please select a valid ONNX model file."); return
        if not output_dir: QMessageBox.critical(self, "Error", "Please select an output directory."); return
        if not classes_str: QMessageBox.critical(self, "Error", "Please enter class names."); return
        try:
            class_list = [name.strip() for name in classes_str.split(',') if name.strip()]
            if not class_list: raise ValueError("No valid classes found.")
        except Exception as e: QMessageBox.critical(self, "Error", f"Invalid class names format: {e}"); return
        input_path = None; camera_id = -1; is_live = False; process_images = False
        if self.radio_camera.isChecked():
            try: camera_id = int(self.cam_id_entry.text()); is_live = True
            except ValueError: QMessageBox.critical(self, "Error", "Invalid Camera ID."); return
        elif self.radio_video.isChecked():
            input_path = self.video_entry.text()
            if not input_path or not os.path.exists(input_path):
                QMessageBox.critical(self, "Error", "Please select a valid video file."); return
        elif self.radio_image_dir.isChecked():
            input_path = self.img_dir_entry.text()
            if not input_path or not os.path.isdir(input_path):
                QMessageBox.critical(self, "Error", "Please select a valid image directory."); return
            process_images = True
        else: QMessageBox.critical(self, "Error", "No input source selected."); return

        # --- Initialize Tracker ---
        try:
            self.update_status("Status: Initializing Tracker...")
            QApplication.processEvents()
            self.tracker_instance = self.ObjectTracker(
                onnx_model_path=model_path, class_names=class_list,
                confidence_threshold=conf, iou_threshold=iou
            )
            print("Tracker initialized successfully in GUI.")
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize tracker: {e}")
            self.update_status("Status: Error"); return

        # --- Create Worker and Submit to Executor ---
        try: os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
             QMessageBox.critical(self, "Error", f"Could not create output directory: {e}")
             self.update_status("Status: Error"); return

        # Create worker instance, passing self (main window) for callbacks via invokeMethod
        self.worker_instance = TrackingWorker(
            main_window=self, # Pass reference to main window
            tracker_instance=self.tracker_instance,
            input_path=input_path, camera_id=camera_id, output_dir=output_dir,
            save_frames=save_frames, is_live=is_live, process_images=process_images
        )

        # Submit the worker's run method to the executor
        self.current_future = self.executor.submit(self.worker_instance.run)
        # Add a callback to handle completion/errors
        self.current_future.add_done_callback(self._tracking_done_callback)

        self.is_tracking = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.update_status("Status: Tracking...")

    @Slot()
    def stop_tracking(self):
        """Signals the worker instance to stop."""
        if self.is_tracking and self.worker_instance:
            print("GUI requesting worker stop...")
            self.worker_instance.stop() # Call the worker's stop method
            # Optionally try to cancel the future, though it might not interrupt running task
            if self.current_future and not self.current_future.done():
                 cancelled = self.current_future.cancel()
                 print(f"Future cancellation attempt result: {cancelled}")
            self.stop_button.setEnabled(False) # Prevent multiple clicks while stopping
        else:
             print("Stop clicked but no active tracking found.")
             self.finalize_tracking_state()


    def _tracking_done_callback(self, future: Future):
        """Callback executed when the future finishes (normally or with exception)."""
        print("Future done callback executed.")
        try:
            exception = future.exception() # Check if an exception occurred in the worker's run method
            if exception:
                print(f"Exception from worker thread future: {exception}")
                # Ensure error is reported via the GUI thread
                QMetaObject.invokeMethod(self, "_handle_error", Qt.ConnectionType.QueuedConnection,
                                         Q_ARG(str, f"Worker error: {exception}"))
            else:
                print("Tracking process completed successfully (according to future).")
                # Ensure status is updated via GUI thread
                QMetaObject.invokeMethod(self, "_handle_progress", Qt.ConnectionType.QueuedConnection,
                                         Q_ARG(str, "Status: Completed"))

        except Exception as e:
             # Error within the callback itself
             print(f"Error in done_callback: {e}")
             QMetaObject.invokeMethod(self, "_handle_error", Qt.ConnectionType.QueuedConnection,
                                      Q_ARG(str, f"Callback error: {e}"))
        finally:
             # Finalize state regardless of success/failure, safely from GUI thread
             QMetaObject.invokeMethod(self, "finalize_tracking_state", Qt.ConnectionType.QueuedConnection)


    # --- Slots Called by Worker via invokeMethod ---
    @Slot(str)
    def _handle_progress(self, status_text):
        """Updates the status bar label from worker."""
        self.update_status(status_text)

    @Slot(QImage) # Expecting QImage now
    def _handle_frame_qimage(self, qt_image):
        """Updates the video display label with a QImage from worker."""
        try:
            if qt_image.isNull():
                print("Received null QImage.")
                return
            # Create QPixmap, scale it to fit the label while maintaining aspect ratio
            qt_pixmap = QPixmap.fromImage(qt_image)
            # Check if label size is valid before scaling
            label_size = self.video_display_label.size()
            if label_size.width() <= 0 or label_size.height() <= 0:
                print(f"Warning: Invalid label size for scaling: {label_size}")
                # Fallback: display original size or skip update
                self.video_display_label.setPixmap(qt_pixmap)
                return

            scaled_pixmap = qt_pixmap.scaled(label_size,
                                             Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)
            self.video_display_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error updating video display from QImage: {e}")

    @Slot(str)
    def _handle_error(self, error_message):
        """Handles errors reported by the worker thread via invokeMethod."""
        print(f"Worker thread error reported via invokeMethod: {error_message}")
        QMessageBox.critical(self, "Tracking Error", error_message)
        # Don't finalize state here, let the done_callback handle it


    @Slot() # Make finalize_tracking_state a slot to be callable via invokeMethod
    def finalize_tracking_state(self):
         """Resets GUI state and cleans up resources. Safe to call from main thread."""
         if not self.is_tracking and not self.start_button.isEnabled():
              print("Finalize called but state seems already reset or inconsistent. Forcing reset.")
         elif not self.is_tracking:
              print("Finalize called but not tracking. No action needed.")
              return # Avoid resetting if already idle

         print("Finalizing tracking state in GUI...")
         self.is_tracking = False
         self.start_button.setEnabled(True)
         self.stop_button.setEnabled(False)
         self.update_status("Status: Idle / Stopped / Completed")
         self.tracker_instance = None
         self.worker_instance = None
         self.current_future = None
         # Reset video display to placeholder
         self.video_display_label.setPixmap(self.placeholder_pixmap)
         print("GUI state finalized.")


    # --- Utility Slots ---
    @Slot(str)
    def update_status(self, status_text):
        """Updates the status bar label directly."""
        self.status_label.setText(status_text)

    # --- Window Close Event ---
    def closeEvent(self, event):
        """Handles the window close event."""
        print("Close event triggered.")
        if self.is_tracking:
            reply = QMessageBox.question(self, 'Confirm Exit',
                                         'Tracking is in progress. Stop tracking and exit?',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                print("Attempting to stop tracking on close...")
                self.stop_tracking() # Request worker stop
                print("Shutting down executor...")
                # Shutdown executor - cancel pending, don't wait for running
                # wait=False allows the GUI to close quickly
                self.executor.shutdown(wait=False, cancel_futures=True)
                print("Executor shutdown requested.")
                event.accept() # Allow window to close
            else:
                event.ignore() # Keep window open
        else:
            print("Shutting down executor (not tracking)...")
            self.executor.shutdown(wait=False) # Still good practice to shutdown
            print("Executor shutdown requested.")
            event.accept() # Allow window to close


# --- Main Execution ---
if __name__ == "__main__":
    # Attempt to import ObjectTracker here again to catch early errors if run directly
    try:
        from object_tracker import ObjectTracker
    except ImportError as e:
         # Use a simple Tkinter fallback for the error if PySide6 isn't fully up yet
         try:
             import tkinter as tk
             from tkinter import messagebox
             root = tk.Tk()
             root.withdraw() # Hide the main Tk window
             messagebox.showerror("Fatal Import Error", f"Could not import ObjectTracker: {e}\nApplication cannot start.")
             root.destroy()
         except ImportError:
             print(f"CRITICAL ERROR: Could not import ObjectTracker: {e}")
             print("Cannot display graphical error message as Tkinter also failed.")
         sys.exit(1)


    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = TrackingAppGUI()

    # Check if ObjectTracker loaded successfully before showing window
    if window.ObjectTracker is None:
         print("ObjectTracker class failed to load. Exiting.")
         sys.exit(1) # Exit if the critical import failed in __init__

    window.show()
    sys.exit(app.exec())
