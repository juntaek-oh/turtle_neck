import collections
import time

import cv2
import numpy as np
import openvino as ov
import openvino.properties.hint as hints
from IPython import display
from numpy.lib.stride_tricks import as_strided

from .open_pose_decoder import OpenPoseDecoder
from .video_utils import VideoPlayer

decoder = OpenPoseDecoder()


class Detector:
    def __init__(self, model_path, device):
        # Initialize OpenVINO Runtime
        core = ov.Core()
        # Read the network from a file.
        model = core.read_model(model_path)
        # Let the AUTO device decide where to load the model (you can use CPU, GPU as well).
        self.compiled_model = core.compile_model(
            model=model,
            device_name=device,
            config={hints.performance_mode(): hints.PerformanceMode.LATENCY},
        )

        # Get the input and output names of nodes.
        input_layer = self.compiled_model.input(0)
        output_layers = self.compiled_model.outputs

        # Get the input size.
        self.height, self.width = list(input_layer.shape)[2:]

        input_layer.any_name, [o.any_name for o in output_layers]

        self.colors = (
            (255, 0, 0),
            (255, 0, 255),
            (170, 0, 255),
            (255, 0, 85),
            (255, 0, 170),
            (85, 255, 0),
            (255, 170, 0),
            (0, 255, 0),
            (255, 255, 0),
            (0, 255, 85),
            (170, 255, 0),
            (0, 85, 255),
            (0, 255, 170),
            (0, 0, 255),
            (0, 255, 255),
            (85, 0, 255),
            (0, 170, 255),
        )

        self.default_skeleton = (
            (15, 13),
            (13, 11),
            (16, 14),
            (14, 12),
            (11, 12),
            (5, 11),
            (6, 12),
            (5, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (1, 2),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
        )
        # --- 자세 보정 및 민감도 설정 ---
        self.CALIBRATION_TIME = 4  # 5초 후에 자세를 측정합니다.
        self.SENSITIVITY = 45  # 기준 자세보다 이 값(픽셀)만큼 더 숙이면 경고합니다.

        # --- 내부 상태 변수 (수정하지 마세요) ---
        self.g_calibration_start_time = time.time()
        self.g_baseline_y_diff = None

    # 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
    def pool2d(self, A, kernel_size, stride, padding, pool_mode="max"):
        """
        2D Pooling

        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        """
        # Padding
        A = np.pad(A, padding, mode="constant")

        # Window view of A
        output_shape = (
            (A.shape[0] - kernel_size) // stride + 1,
            (A.shape[1] - kernel_size) // stride + 1,
        )
        kernel_size = (kernel_size, kernel_size)
        A_w = as_strided(
            A,
            shape=output_shape + kernel_size,
            strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
        )
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling.
        if pool_mode == "max":
            return A_w.max(axis=(1, 2)).reshape(output_shape)
        elif pool_mode == "avg":
            return A_w.mean(axis=(1, 2)).reshape(output_shape)

    # non maximum suppression
    def heatmap_nms(self, heatmaps, pooled_heatmaps):
        return heatmaps * (heatmaps == pooled_heatmaps)

    def process_results(self, img, pafs, heatmaps):
        # This processing comes from
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
        pooled_heatmaps = np.array(
            [
                [
                    self.pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max")
                    for h in heatmaps[0]
                ]
            ]
        )
        nms_heatmaps = self.heatmap_nms(heatmaps, pooled_heatmaps)

        # Decode poses.
        poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
        output_shape = list(self.compiled_model.output(index=0).partial_shape)
        output_scale = (
            img.shape[1] / output_shape[3].get_length(),
            img.shape[0] / output_shape[2].get_length(),
        )
        # Multiply coordinates by a scaling factor.
        poses[:, :, :2] *= output_scale
        return poses, scores

    def draw_poses(self, img, poses, point_score_threshold):

        if poses.size == 0 and self.g_baseline_y_diff is None:
            # 화면에 사람이 없으면 보정 진행이 안됨을 알림
            cv2.putText(
                img,
                "Please be in front of the camera.",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            return img

        # 1. 기준점이 아직 설정되지 않았을 경우 (보정 및 캡처 단계)
        if self.g_baseline_y_diff is None:
            elapsed_time = time.time() - self.g_calibration_start_time

            # 1-1. 5초 카운트다운
            if elapsed_time <= self.CALIBRATION_TIME:
                status_text = (
                    f"Get Ready... {self.CALIBRATION_TIME - elapsed_time:.1f}s"
                )
                cv2.putText(
                    img,
                    status_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                )
                cv2.putText(
                    img,
                    "At 0 sec, hold your BEST posture!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            # 1-2. 5초 후, 기준점 캡처
            else:
                capture_success = False
                for pose in poses:
                    nose_score = pose[0, 2]
                    l_shoulder_score = pose[5, 2]
                    r_shoulder_score = pose[6, 2]

                    if (
                        nose_score > point_score_threshold
                        and l_shoulder_score > point_score_threshold
                        and r_shoulder_score > point_score_threshold
                    ):
                        # 유효한 자세가 감지되면, 현재 y좌표 차이를 기준점으로 설정
                        nose_y = pose[0, 1]
                        l_shoulder_y = pose[5, 1]
                        r_shoulder_y = pose[6, 1]
                        shoulder_y_avg = (l_shoulder_y + r_shoulder_y) / 2

                        self.g_baseline_y_diff = nose_y - shoulder_y_avg

                        print(f"Baseline Captured! Value: {self.g_baseline_y_diff:.2f}")
                        cv2.putText(
                            img,
                            "Baseline CAPTURED!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 0),
                            3,
                        )
                        capture_success = True
                        break  # 첫 번째 감지된 자세로 기준을 잡고 반복 중단

                if not capture_success:
                    cv2.putText(
                        img,
                        "Looking for you... Hold posture!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

        # 2. 기준점이 설정된 후 (모니터링 단계)
        else:
            for pose in poses:
                nose_score = pose[0, 2]
                l_shoulder_score = pose[5, 2]
                r_shoulder_score = pose[6, 2]

                if (
                    nose_score > point_score_threshold
                    and l_shoulder_score > point_score_threshold
                    and r_shoulder_score > point_score_threshold
                ):
                    nose_y = pose[0, 1]
                    l_shoulder_y = pose[5, 1]
                    r_shoulder_y = pose[6, 1]
                    shoulder_y_avg = (l_shoulder_y + r_shoulder_y) / 2
                    current_y_diff = nose_y - shoulder_y_avg

                    cv2.putText(
                        img,
                        f"Baseline: {self.g_baseline_y_diff:.1f}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        img,
                        f"Current: {current_y_diff:.1f}",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    # 거북목 판단 로직
                    if current_y_diff > self.g_baseline_y_diff + self.SENSITIVITY:
                        cv2.putText(
                            img,
                            "WARNING: Turtle Neck!",
                            (100, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            3,
                        )

        # 자세 시각화 (기존과 동일)
        img_limbs = np.copy(img)
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > point_score_threshold:
                    cv2.circle(img, tuple(p), 1, self.colors[i], 2)
            for i, j in self.default_skeleton:
                if (
                    points_scores[i] > point_score_threshold
                    and points_scores[j] > point_score_threshold
                ):
                    cv2.line(
                        img_limbs,
                        tuple(points[i]),
                        tuple(points[j]),
                        color=self.colors[j],
                        thickness=4,
                    )
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img

    # --------------------------------------------------------------------------

    # Main processing function to run pose estimation.
    def run_pose_estimation(
        self, source=0, flip=False, use_popup=False, skip_first_frames=0
    ):
        pafs_output_key = self.compiled_model.output("Mconv7_stage2_L1")
        heatmaps_output_key = self.compiled_model.output("Mconv7_stage2_L2")
        player = None
        try:
            # Create a video player to play with target fps.
            player = VideoPlayer(
                source, flip=flip, fps=30, skip_first_frames=skip_first_frames
            )
            # Start capturing.
            player.start()
            if use_popup:
                title = "Press ESC to Exit"
                cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

            processing_times = collections.deque()

            while True:
                # Grab the frame.
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break

                # If the frame is larger than full HD, reduce size to improve the performance.
                scale = 1280 / max(frame.shape)
                if scale < 1:
                    frame = cv2.resize(
                        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )

                # Resize the image and change dims to fit neural network input.
                # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
                input_img = cv2.resize(
                    frame, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
                # Create a batch of images (size = 1).
                input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

                # Measure processing time.
                start_time = time.time()
                # Get results.
                results = self.compiled_model([input_img])
                stop_time = time.time()

                pafs = results[pafs_output_key]
                heatmaps = results[heatmaps_output_key]
                # Get poses from network results.
                poses, scores = self.process_results(frame, pafs, heatmaps)

                # Draw poses on a frame.
                frame = self.draw_poses(frame, poses, 0.1)

                processing_times.append(stop_time - start_time)
                # Use processing times from last 200 frames.
                if len(processing_times) > 200:
                    processing_times.popleft()

                _, f_width = frame.shape[:2]
                # mean processing time [ms]
                processing_time = np.mean(processing_times) * 1000
                fps = 1000 / processing_time
                cv2.putText(
                    frame,
                    f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    (20, 40),
                    cv2.FONT_HERSHEY_COMPLEX,
                    f_width / 1000,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Use this workaround if there is flickering.
                if use_popup:
                    cv2.imshow(title, frame)
                    key = cv2.waitKey(1)
                    # escape = 27
                    if key == 27:
                        break
                else:
                    # Encode numpy array to jpg.
                    _, encoded_img = cv2.imencode(
                        ".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                    # Create an IPython image.
                    i = display.Image(data=encoded_img)
                    # Display the image in this notebook.
                    display.clear_output(wait=True)
                    display.display(i)
        # ctrl-c
        except KeyboardInterrupt:
            print("Interrupted")
        # any different error
        except RuntimeError as e:
            print(e)
        finally:
            if player is not None:
                # Stop capturing.
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()
