from model_factory.interfaces.yolo import ObjectYoloInterface

class PeopleDetectorModel(ObjectYoloInterface):
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name
        # Inicialização e configuração do modelo YOLOv8 aqui

    def generate_conf(self):
        print("YOLO!!!", "Generate conf...")
        pass

    def generate_data(self):
        print("YOLO!!!", "Generate data...")
        pass

    def generate_model(self):
        print("YOLO!!!", "Generate model...")
        pass

    def train_model(self):
        print("YOLO!!!", "Training...")
        pass

    def export_model(self):
        print("YOLO!!!", "Exporting data...")
        pass
    
    def detect_and_predict(self):
        print("YOLO!!!", "Detecting...")
        pass

