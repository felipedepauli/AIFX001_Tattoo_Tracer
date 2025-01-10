from model_factory.interfaces.classification import ObjectClassificationInterface
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model_factory.nets.net_sample import create_tattoo_classifier_model

class TattooClassifierModel(ObjectClassificationInterface):
    def __init__(self, config, model_name):
        self.config = config
        self.model = None
        self.model_name = model_name


    def generate_conf(self):
        pass

    def generate_data(self, train=True):
        # Data augmentation settings based on the YAML file
        if self.config['data_augmentation']['enabled'] and train:
            data_gen_args = {
                'rotation_range': self.config['data_augmentation']['rotation_range'],
                'width_shift_range': self.config['data_augmentation']['width_shift_range'],
                'height_shift_range': self.config['data_augmentation']['height_shift_range'],
                'shear_range': self.config['data_augmentation']['shear_range'],
                'zoom_range': self.config['data_augmentation']['zoom_range'],
                'horizontal_flip': self.config['data_augmentation']['horizontal_flip'],
                'fill_mode': 'nearest'
            }
        else:
            data_gen_args = {}

        # Instantiate the ImageDataGenerator with the data augmentation settings
        datagen = ImageDataGenerator(**data_gen_args)

        # Path of the image directory based on training or validation
        if train:
            data_dir = self.config['train_path']
        else:
            data_dir = self.config['val_path']

        # Creating the data generator
        data_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(150, 150),  # Assuming you want to resize the images
            batch_size=self.config['batch_size'],
            class_mode='binary'  # Assuming binary classification
        )

        return data_generator
    
    # Isso vai pra nets/cnn
    def generate_model(self):
        self.model = create_tattoo_classifier_model()

    def train_model(self):
        print(">>>>>>>>>>> TattooClassifierModel's fit method")
        # Assuming you have functions or methods that return the data generators:
        train_generator      = self.generate_data(train=True)
        validation_generator = self.generate_data(train=False)

        # Settings extracted from the YAML file
        epochs = self.config['epochs']
        steps_per_epoch = self.config['steps_per_epoch']
        validation_steps = self.config.get('validation_steps', steps_per_epoch // 10)  # Example calculation
        batch_size = self.config['batch_size']
        early_stop = self.config['early_stop']

        # Configuring EarlyStopping if necessary
        callbacks = []
        if early_stop:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=3))

        # Configuring ModelCheckpoint to save the best model
        model_checkpoint_callback = ModelCheckpoint(
            filepath=f'./models/{self.model_name}.h5',  # Specify the path to save the best model
            save_best_only=True,  # Save only the best model
            monitor='val_loss',  # Monitor the validation loss
            mode='min',  # The best model will have the minimum validation loss
            verbose=1  # Log a message when a better model is found and saved
        )
        callbacks.append(model_checkpoint_callback)

        # Call to the fit method
        self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

    def export_model(self):
        pass
    
    def predict(self):
        pass
    
    def evaluate_model(self):
        pass