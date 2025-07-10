from turtle_neck import Detector, DownloadHelper
from pathlib import Path

# A directory where the model will be downloaded.
base_model_dir = Path("model")
# The name of the model from Open Model Zoo.
model_name = "human-pose-estimation-0001"
# Selected precision (FP32, FP16, FP16-INT8).
precision = "FP16-INT8"

model_path = base_model_dir / "intel" / model_name / precision / f"{model_name}.xml"

download_helper = DownloadHelper()

if not model_path.exists():
    model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
    download_helper.download_file(
        model_url_dir + model_name + ".xml", model_path.name, model_path.parent
    )
    download_helper.download_file(
        model_url_dir + model_name + ".bin",
        model_path.with_suffix(".bin").name,
        model_path.parent,
    )

detector = Detector(model_path=model_path, device="CPU")

USE_WEBCAM = True
cam_id = 0
video_file = Path("store-aisle-detection.mp4")
video_url = "https://storage.openvinotoolkit.org/data/test_data/videos/store-aisle-detection.mp4"
source = cam_id if USE_WEBCAM else video_file


additional_options = {"skip_first_frames": 500} if not USE_WEBCAM else {}
detector.run_pose_estimation(
    source=source, flip=isinstance(source, int), use_popup=True, **additional_options
)
