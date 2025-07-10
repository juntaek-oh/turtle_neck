"""
자세 시각화 모듈
"""
import numpy as np
import cv2


# 색상 정의 (각 키포인트별)
COLORS = (
    (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170),
    (85, 255, 0), (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85),
    (170, 255, 0), (0, 85, 255), (0, 255, 170), (0, 0, 255), (0, 255, 255),
    (85, 0, 255), (0, 170, 255),
)

# 기본 스켈레톤 구조 (COCO 포맷)
DEFAULT_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
)


class PoseVisualizer:
    """자세 시각화를 담당하는 클래스"""
    
    def __init__(self, colors=COLORS, skeleton=DEFAULT_SKELETON):
        self.colors = colors
        self.skeleton = skeleton
    
    def draw_keypoints(self, img, poses, point_score_threshold=0.1):
        """키포인트 그리기"""
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            
            # 키포인트 그리기
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > point_score_threshold:
                    cv2.circle(img, tuple(p), 3, self.colors[i], 2)
        
        return img
    
    def draw_skeleton(self, img, poses, point_score_threshold=0.1, line_thickness=4):
        """스켈레톤 그리기"""
        img_limbs = np.copy(img)
        
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            
            # 스켈레톤 라인 그리기
            for i, j in self.skeleton:
                if (points_scores[i] > point_score_threshold and 
                    points_scores[j] > point_score_threshold):
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), 
                            color=self.colors[j], thickness=line_thickness)
        
        # 원본 이미지와 스켈레톤 이미지 블렌딩
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img
    
    def draw_poses(self, img, poses, point_score_threshold=0.1):
        """전체 자세 그리기 (키포인트 + 스켈레톤)"""
        # 키포인트 그리기
        img = self.draw_keypoints(img, poses, point_score_threshold)
        
        # 스켈레톤 그리기
        img = self.draw_skeleton(img, poses, point_score_threshold)
        
        return img
    
    def draw_fps_info(self, img, processing_time):
        """FPS 정보 그리기"""
        _, f_width = img.shape[:2]
        fps = 1000 / processing_time if processing_time > 0 else 0
        
        cv2.putText(
            img,
            f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            (20, 40),
            cv2.FONT_HERSHEY_COMPLEX,
            f_width / 1000,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return img
    
    def highlight_neck_keypoints(self, img, poses, point_score_threshold=0.1):
        """목 관련 키포인트 강조 표시"""
        NOSE_IDX = 0
        LEFT_SHOULDER_IDX = 5
        RIGHT_SHOULDER_IDX = 6
        
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            
            # 코 강조
            if points_scores[NOSE_IDX] > point_score_threshold:
                cv2.circle(img, tuple(points[NOSE_IDX]), 8, (0, 255, 255), -1)
                cv2.putText(img, "NOSE", (points[NOSE_IDX][0] + 10, points[NOSE_IDX][1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 어깨 강조
            if points_scores[LEFT_SHOULDER_IDX] > point_score_threshold:
                cv2.circle(img, tuple(points[LEFT_SHOULDER_IDX]), 8, (255, 255, 0), -1)
            
            if points_scores[RIGHT_SHOULDER_IDX] > point_score_threshold:
                cv2.circle(img, tuple(points[RIGHT_SHOULDER_IDX]), 8, (255, 255, 0), -1)
            
            # 어깨 중심점 표시
            if (points_scores[LEFT_SHOULDER_IDX] > point_score_threshold and 
                points_scores[RIGHT_SHOULDER_IDX] > point_score_threshold):
                shoulder_center = ((points[LEFT_SHOULDER_IDX] + points[RIGHT_SHOULDER_IDX]) // 2)
                cv2.circle(img, tuple(shoulder_center), 6, (0, 255, 0), -1)
                cv2.putText(img, "SHOULDER", (shoulder_center[0] + 10, shoulder_center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img
    
    def draw_measurement_line(self, img, poses, point_score_threshold=0.1):
        """측정 라인 그리기 (코와 어깨 중심을 연결)"""
        NOSE_IDX = 0
        LEFT_SHOULDER_IDX = 5
        RIGHT_SHOULDER_IDX = 6
        
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            
            if (points_scores[NOSE_IDX] > point_score_threshold and 
                points_scores[LEFT_SHOULDER_IDX] > point_score_threshold and 
                points_scores[RIGHT_SHOULDER_IDX] > point_score_threshold):
                
                nose_point = tuple(points[NOSE_IDX])
                shoulder_center = tuple((points[LEFT_SHOULDER_IDX] + points[RIGHT_SHOULDER_IDX]) // 2)
                
                # 측정 라인 그리기
                cv2.line(img, nose_point, shoulder_center, (255, 0, 255), 3)
                
                # 수직 기준선 그리기
                vertical_line_start = (shoulder_center[0], shoulder_center[1] - 100)
                vertical_line_end = (shoulder_center[0], shoulder_center[1] + 100)
                cv2.line(img, vertical_line_start, vertical_line_end, (0, 255, 0), 2)
        
        return img