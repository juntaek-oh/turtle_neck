"""
모델 다운로드, 초기화, 컴파일을 담당하는 모듈
"""
from pathlib import Path
import openvino as ov
import openvino.properties.hint as hints
import requests


class ModelManager:
    def __init__(self, model_name="human-pose-estimation-0001", precision="FP16-INT8"):
        self.model_name = model_name
        self.precision = precision
        self.base_model_dir = Path("model")
        self.model_path = self.base_model_dir / "intel" / model_name / precision / f"{model_name}.xml"
        self.core = ov.Core()
        self.model = None
        self.compiled_model = None
        
    def download_notebook_utils(self):
        """notebook_utils 모듈 다운로드"""
        if not Path("notebook_utils.py").exists():
            r = requests.get(
                url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
            )
            open("notebook_utils.py", "w").write(r.text)
        
        import notebook_utils as utils
        return utils
    
    def download_model(self):
        """모델 파일 다운로드"""
        if not self.model_path.exists():
            utils = self.download_notebook_utils()
            model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{self.model_name}/{self.precision}/"
            utils.download_file(
                model_url_dir + self.model_name + ".xml", 
                self.model_path.name, 
                self.model_path.parent
            )
            utils.download_file(
                model_url_dir + self.model_name + ".bin",
                self.model_path.with_suffix(".bin").name,
                self.model_path.parent,
            )
    
    def load_model(self, device="AUTO"):
        """모델 로드 및 컴파일"""
        self.download_model()
        
        # 모델 읽기
        self.model = self.core.read_model(self.model_path)
        
        # 모델 컴파일
        self.compiled_model = self.core.compile_model(
            model=self.model, 
            device_name=device, 
            config={hints.performance_mode(): hints.PerformanceMode.LATENCY}
        )
        
        return self.compiled_model
    
    def get_model_info(self):
        """모델 입출력 정보 반환"""
        if self.compiled_model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 실행하세요.")
        
        input_layer = self.compiled_model.input(0)
        output_layers = self.compiled_model.outputs
        
        # 입력 크기 추출
        height, width = list(input_layer.shape)[2:]
        
        return {
            "input_layer": input_layer,
            "output_layers": output_layers,
            "height": height,
            "width": width,
            "input_name": input_layer.any_name,
            "output_names": [o.any_name for o in output_layers]
        }