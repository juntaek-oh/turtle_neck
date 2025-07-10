"""
이미지 전처리, 후처리 및 유틸리티 함수들
"""
import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided


def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling in numpy
    
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

    # Return the result of pooling
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


def heatmap_nms(heatmaps, pooled_heatmaps):
    """Non Maximum Suppression for heatmaps"""
    return heatmaps * (heatmaps == pooled_heatmaps)


class ImageProcessor:
    """이미지 전처리 및 후처리를 담당하는 클래스"""
    
    def __init__(self, input_width, input_height):
        self.input_width = input_width
        self.input_height = input_height
    
    def preprocess_frame(self, frame):
        """프레임 전처리"""
        # 해상도가 너무 크면 성능 향상을 위해 크기 조정
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # 신경망 입력에 맞게 크기 조정 및 차원 변경
        input_img = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]
        
        return frame, input_img
    
    def process_results(self, img, pafs, heatmaps, decoder, compiled_model):
        """모델 결과 후처리"""
        # Max pooling 적용
        pooled_heatmaps = np.array([[
            pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") 
            for h in heatmaps[0]
        ]])
        
        # NMS 적용
        nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

        # 자세 디코딩
        poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
        
        # 출력 스케일 계산
        output_shape = list(compiled_model.output(index=0).partial_shape)
        output_scale = (
            img.shape[1] / output_shape[3].get_length(),
            img.shape[0] / output_shape[2].get_length(),
        )
        
        # 좌표에 스케일링 팩터 적용
        poses[:, :, :2] *= output_scale
        
        return poses, scores
    
    @staticmethod
    def resize_frame_if_needed(frame, max_dimension=1280):
        """필요시 프레임 크기 조정"""
        scale = max_dimension / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return frame
    
    @staticmethod
    def prepare_input_image(frame, target_width, target_height):
        """입력 이미지 준비"""
        input_img = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]
        return input_img