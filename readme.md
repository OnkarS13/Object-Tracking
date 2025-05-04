python run_tracking.py --model ./best.onnx --camera_id 0 --output_dir ./tracking_output_live --save_frames --classes car,motorbike,person

python run_tracking.py --model ./best.onnx --camera_id 1 --output_dir ./tracking_output_live --save_frames --classes car,motorbike,person

python run_tracking.py --model ./best.onnx --video "0000f77c-6257be58.mov" --output_dir ./tracking_output_live --save_frames --classes car,motorbike,person
