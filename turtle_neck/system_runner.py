"""
시스템 실행 및 관리 모듈
"""
import collections
import time
from pathlib import Path
import cv2
import numpy as np
from IPython import display

from .model_manager import ModelManager
from .pose_decoder import OpenPoseDecoder
from .image_processor import ImageProcessor
from .turtle_neck_detector import TurtleNeckDetector
from .pose_visualizer import PoseVisualizer


class TurtleNeckSystem:
    """거북목 감지 시스템 메인 클래스"""
    
    def __init__(self, calibration_time=4, sensitivity=45, device="AUTO"):
        # 각 모듈 초기화
        self.model_manager = ModelManager()
        self.decoder = OpenPoseDecoder()
        self.turtle_detector = TurtleNeckDetector(calibration_time, sensitivity)
        self.visualizer = PoseVisualizer()
        
        # 모델 로드
        print("Loading model...")
        self.compiled_model = self.model_manager.load_model(device)
        model_info = self.model_manager.get_model_info()
        
        # 이미지 프로세서 초기화
        self.image_processor = ImageProcessor(model_info["width"], model_info["height"])
        
        # 출력 키 설정
        self.pafs_output_key = self.compiled_model.output("Mconv7_stage2_L1")
        self.heatmaps_output_key = self.compiled_model.output("Mconv7_stage2_L2")
        
        print("System initialization complete!")
    
    def process_frame(self, frame):
        """단일 프레임 처리"""
        # 전처리
        processed_frame, input_img = self.image_processor.preprocess_frame(frame)
        
        # 모델 추론
        results = self.compiled_model([input_img])
        pafs = results[self.pafs_output_key]
        heatmaps = results[self.heatmaps_output_key]
        
        # 자세 추출
        poses, scores = self.image_processor.process_results(
            processed_frame, pafs, heatmaps, self.decoder, self.compiled_model
        )
        
        return processed_frame, poses, scores
    
    def run_webcam(self, source=0, flip=False, use_popup=False, skip_first_frames=0):
        """웹캠으로 실시간 거북목 감지 실행"""
        # notebook_utils import
        import sys
        if Path("notebook_utils.py").exists():
            import notebook_utils as utils
        else:
            utils = self.model_manager.download_notebook_utils()
        
        player = None
        try:
            # 비디오 플레이어 생성
            player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
            player.start()
            
            if use_popup:
                title = "Turtle Neck Detection - Press ESC to Exit"
                cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

            processing_times = collections.deque()

            while True:
                # 프레임 캡처
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break

                # 프레임 처리 시작 시간
                start_time = time.time()
                
                # 프레임 처리
                processed_frame, poses, scores = self.process_frame(frame)
                
                # 거북목 감지 및 상태 정보 그리기
                processed_frame = self.turtle_detector.draw_status_info(processed_frame, poses)
                
                # 자세 시각화
                processed_frame = self.visualizer.draw_poses(processed_frame, poses, 0.1)
                
                # 처리 시간 계산 (FPS 표시용이 아닌 성능 모니터링용으로만 사용)
                stop_time = time.time()
                
                # 처리 시간 기록 (내부적으로만 사용)
                processing_times.append(stop_time - start_time)
                if len(processing_times) > 200:
                    processing_times.popleft()

                # 화면 출력
                if use_popup:
                    cv2.imshow(title, processed_frame)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC 키
                        break
                    elif key == ord('r'):  # R 키로 보정 재시작
                        self.turtle_detector.reset_calibration()
                else:
                    # Jupyter notebook에서 출력
                    _, encoded_img = cv2.imencode(".jpg", processed_frame, 
                                                params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                    i = display.Image(data=encoded_img)
                    display.clear_output(wait=True)
                    display.display(i)

        except KeyboardInterrupt:
            print("Interrupted")
        except RuntimeError as e:
            print(e)
        finally:
            if player is not None:
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()
    
    def reset_calibration(self):
        """보정 재시작"""
        self.turtle_detector.reset_calibration()
    
    def get_calibration_status(self):
        """보정 상태 확인"""
        return self.turtle_detector.is_calibrated()


def run_turtle_neck_detection():
    """Run turtle neck detection system"""
    # Initialize turtle neck detection system
    system = TurtleNeckSystem(calibration_time=4, sensitivity=45)
    
    print("=== Turtle Neck Detection System ===")
    print("Controls:")
    print("- ESC: Exit")
    print("- R: Reset calibration")
    print("- Hold your best posture after 4 seconds!")
    print("=====================================")
    
    # Run with webcam
    system.run_webcam(source=0, flip=True, use_popup=True)