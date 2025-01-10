import json

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

from model_factory.nets.rcnn.tattoo_detect import create_tattoo_detection_model
from model_factory.interfaces.detection import ObjectDetectionInterfaces

class TattooDetectorModel(ObjectDetectionInterfaces):
    def __init__(self, config, model_name):
        self.config             = config
        self.model_name         = model_name
        self.model              = None
        self.train_generator    = None
        self.valid_generator    = None

    def generate_conf(self):
        print("Tattoo Detector Model!!!", "Generating configuration...")

    def generate_data(self):
        print("Tattoo Detector Model!!!", "Generating data...")
        directory = self.config.get("full_path", '')
        img_size  = self.config.get("img_size", 224)

        train_datagen = ImageDataGenerator(
            rescale             = 1./255,
            rotation_range      = self.config.get("data_augmentation", {}).get("rotation_range", 20),
            width_shift_range   = self.config.get("data_augmentation", {}).get("width_shift_range", 0.2),
            height_shift_range  = self.config.get("data_augmentation", {}).get("height_shift_range", 0.2),
            shear_range         = self.config.get("data_augmentation", {}).get("shear_range", 0.2),
            zoom_range          = self.config.get("data_augmentation", {}).get("zoom_range", 0.2),
            horizontal_flip     = self.config.get("data_augmentation", {}).get("horizontal_flip", True),
            fill_mode           = 'nearest',
            validation_split    = 0.2
        )

        self.train_generator = train_datagen.flow_from_directory(
            directory,
            target_size = (img_size, img_size),
            batch_size  = self.config.get("batch_size", 32),
            class_mode  = 'binary',
            subset      = 'training'
        )

        self.validation_generator = train_datagen.flow_from_directory(
            directory,
            target_size = (img_size, img_size),
            batch_size  = self.config.get("batch_size", 32),
            class_mode  = 'binary',
            subset      = 'validation'
        )

    def generate_model(self):
        print("Tattoo Detector Model!!!", "Generating model...")
        self.model = create_tattoo_detection_model()
        self.model.summary()
        self.model.compile(optimizer=Adam(learning_rate=self.config.get("learning_rate", 0.000001)), loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self):
        print("Tattoo Detector Model!!!", "Training...")
        # Ensure data is generated
        if self.train_generator is None or self.validation_generator is None:
            self.generate_data()

        callbacks = [ModelCheckpoint(filepath=f'./models/{self.model_name}.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)]
        if self.config.get("early_stop", False):
            callbacks.append(EarlyStopping(monitor='val_loss', patience=3))

        history = self.model.fit(
            self.train_generator,
            steps_per_epoch     = self.train_generator.samples // self.config.get("batch_size", 32),
            validation_data     = self.validation_generator,
            validation_steps    = self.validation_generator.samples // self.config.get("batch_size", 32),
            epochs              = self.config.get("epochs", 20),
            callbacks           = callbacks
        )

        # Convert and save training history
        history_dict = history.history
        json_history = json.dumps(history_dict, indent=4)
        with open("model_history.json", "w") as json_file:
            json_file.write(json_history)

    def export_model(self):
        print("Tattoo Detector Model!!!", "Exporting data...")

    def detect(self):
        print("Tattoo Detector Model!!!", "Detecting...")
