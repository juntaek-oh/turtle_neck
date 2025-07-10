"""
거북목 감지 및 상태 관리 모듈
"""
import time
import cv2


class TurtleNeckDetector:
    """거북목 감지 및 상태 관리 클래스"""
    
    def __init__(self, calibration_time=4, sensitivity=45):
        """
        Args:
            calibration_time: 보정에 필요한 시간 (초)
            sensitivity: 민감도 (픽셀 단위)
        """
        self.calibration_time = calibration_time
        self.sensitivity = sensitivity
        
        # 내부 상태 변수
        self.calibration_start_time = time.time()
        self.baseline_y_diff = None
        
        # 키포인트 인덱스 (COCO 포맷)
        self.NOSE_IDX = 0
        self.LEFT_SHOULDER_IDX = 5
        self.RIGHT_SHOULDER_IDX = 6
    
    def reset_calibration(self):
        """Reset calibration"""
        self.calibration_start_time = time.time()
        self.baseline_y_diff = None
        print("Calibration has been reset.")
    
    def is_calibrated(self):
        """Check if calibration is completed"""
        return self.baseline_y_diff is not None
    
    def get_calibration_time_remaining(self):
        """Get remaining time for calibration"""
        if self.is_calibrated():
            return 0
        elapsed = time.time() - self.calibration_start_time
        return max(0, self.calibration_time - elapsed)
    
    def is_valid_pose(self, pose, point_score_threshold=0.1):
        """Check if the pose is valid"""
        nose_score = pose[self.NOSE_IDX, 2]
        l_shoulder_score = pose[self.LEFT_SHOULDER_IDX, 2]
        r_shoulder_score = pose[self.RIGHT_SHOULDER_IDX, 2]
        
        return (nose_score > point_score_threshold and 
                l_shoulder_score > point_score_threshold and 
                r_shoulder_score > point_score_threshold)
    
    def calculate_neck_angle(self, pose):
        """Calculate neck angle (Y coordinate difference between nose and shoulder center)"""
        nose_y = pose[self.NOSE_IDX, 1]
        l_shoulder_y = pose[self.LEFT_SHOULDER_IDX, 1]
        r_shoulder_y = pose[self.RIGHT_SHOULDER_IDX, 1]
        shoulder_y_avg = (l_shoulder_y + r_shoulder_y) / 2
        
        return nose_y - shoulder_y_avg
    
    def attempt_calibration(self, poses, point_score_threshold=0.1):
        """Attempt calibration"""
        for pose in poses:
            if self.is_valid_pose(pose, point_score_threshold):
                current_y_diff = self.calculate_neck_angle(pose)
                self.baseline_y_diff = current_y_diff
                print(f"Baseline captured! Value: {self.baseline_y_diff:.2f}")
                return True
        return False
    
    def detect_turtle_neck(self, poses, point_score_threshold=0.1):
        """Detect turtle neck posture"""
        if not self.is_calibrated():
            return None, "Calibration not completed."
        
        for pose in poses:
            if self.is_valid_pose(pose, point_score_threshold):
                current_y_diff = self.calculate_neck_angle(pose)
                
                # Turtle neck detection
                if current_y_diff > self.baseline_y_diff + self.sensitivity:
                    return True, {
                        "baseline": self.baseline_y_diff,
                        "current": current_y_diff,
                        "difference": current_y_diff - self.baseline_y_diff
                    }
                else:
                    return False, {
                        "baseline": self.baseline_y_diff,
                        "current": current_y_diff,
                        "difference": current_y_diff - self.baseline_y_diff
                    }
        
        return None, "No valid posture detected."
    
    def get_status_message(self, poses, point_score_threshold=0.1):
        """Return status message"""
        if poses.size == 0:
            return "Please stand in front of camera.", (0, 255, 255)
        
        if not self.is_calibrated():
            remaining_time = self.get_calibration_time_remaining()
            
            if remaining_time > 0:
                return f"Get ready... {remaining_time:.1f}s", (0, 255, 0)
            else:
                # Attempt calibration
                if self.attempt_calibration(poses, point_score_threshold):
                    return "Baseline captured!", (0, 255, 0)
                else:
                    return "Please hold your posture!", (0, 255, 255)
        else:
            # Detect turtle neck
            is_turtle_neck, info = self.detect_turtle_neck(poses, point_score_threshold)
            
            if is_turtle_neck is None:
                return "Cannot detect posture.", (0, 255, 255)
            elif is_turtle_neck:
                return "WARNING: Turtle Neck!", (0, 0, 255)
            else:
                return "Good posture!", (0, 255, 0)
    
    def draw_status_info(self, img, poses, point_score_threshold=0.1):
        """Draw status information on image"""
        status_msg, color = self.get_status_message(poses, point_score_threshold)
        
        # Main status message
        cv2.putText(img, status_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Calibration step guidance
        if not self.is_calibrated():
            remaining_time = self.get_calibration_time_remaining()
            if remaining_time > 0:
                cv2.putText(img, "Hold your BEST posture at 0 sec!", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Display detailed information
            is_turtle_neck, info = self.detect_turtle_neck(poses, point_score_threshold)
            if isinstance(info, dict):
                cv2.putText(img, f"Baseline: {info['baseline']:.1f}", (50, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Current: {info['current']:.1f}", (50, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if is_turtle_neck:
                    cv2.putText(img, "WARNING: Turtle Neck!", (100, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        return img