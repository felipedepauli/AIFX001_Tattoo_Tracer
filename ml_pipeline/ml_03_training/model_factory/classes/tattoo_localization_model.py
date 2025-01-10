import os
import random
import matplotlib.pyplot as plt
import numpy as np
from xml.etree import ElementTree
from sklearn.model_selection import train_test_split
from model_factory.interfaces.localization import ObjectLocaterInterface
from model_factory.nets.rcnn.tattoo_loc import create_tattoo_loc_model
from mrcnn.utils import Dataset
import cv2

class TattooLocaterModel(ObjectLocaterInterface):
    """A class to represent a tattoo localization model."""

    def __init__(self, config, model_name):
        """
        Initialize the class with the given configuration and model name.

        Args:
            config (dict): Configuration parameters for the model and training.
            model_name (str): The name of the model for identification.
        """
        self.config                 = config
        self.model_name             = model_name
        self.model                  = None
        self.augmentation_config    = None
        self.dataset_train          = None
        self.dataset_val            = None

    def generate_conf(self):
        """
        Parse and set up the model configuration from the provided config.

        This method prepares the training and data augmentation configurations
        necessary for training the Mask R-CNN model.
        """
        print("Tattoo Localization!!", "Generate conf...")

        # Training configuration
        self.learning_rate   = self.config.get('learning_rate', 0.0001)
        self.epochs          = self.config.get('epochs', 10)
        self.batch_size      = self.config.get('batch_size', 4)
        self.steps_per_epoch = self.config.get('steps_per_epoch', None)

        # Data augmentation configuration
        if self.config.get('data_augmentation', {}).get('enabled', False):
            self.augmentation_config = {
                'rotation_range'    : self.config.get('data_augmentation', {}).get('rotation_range', None),
                'width_shift_range' : self.config.get('data_augmentation', {}).get('width_shift_range', None),
                'height_shift_range': self.config.get('data_augmentation', {}).get('height_shift_range', None),
                'shear_range'       : self.config.get('data_augmentation', {}).get('shear_range', None),
                'zoom_range'        : self.config.get('data_augmentation', {}).get('zoom_range', None),
                'horizontal_flip'   : self.config.get('data_augmentation', {}).get('horizontal_flip', None),
            }
        else:
            self.augmentation_config = None

        print(f"Configuration for '{self.model_name}' generated. Training can be started.")

    def generate_data(self):
        """
        Generate the dataset based on the XML annotations and images.
        """
        print("Tattoo Localization!!", "Generate data...")

        # Define the TattooDataset class inside the generate_data method
        class TattooDataset(Dataset):
            """A class to represent a tattoo dataset."""

            def load_dataset(self, images_dir, annotations_dir, filenames):
                """Load the dataset from the given directories and filenames."""
                self.add_class("dataset", 1, "tattoo")
                count = 0
                for filename in filenames:
                    if count > 100:
                        break
                    count += 1
                    image_id, _ = os.path.splitext(filename)
                    img_path = os.path.join(images_dir, filename)
                    ann_path = os.path.join(annotations_dir, image_id + '.xml')
                    if os.path.exists(ann_path):
                        self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
                        print("Adding image:", os.path.basename(img_path), "with annotation:", os.path.basename(ann_path))

            def extract_boxes(self, filename):
                """Load and parse the file to extract bounding boxes and image dimensions."""
                # Load and parse the XML file
                tree = ElementTree.parse(filename)
                root = tree.getroot()

                # Extract image dimensions
                width  = int(root.find('.//size/width').text)
                height = int(root.find('.//size/height').text)

                # Extract each bounding box
                boxes = []
                for box in root.findall('.//object/bndbox'):
                    xmin = int(box.find('xmin').text)
                    ymin = int(box.find('ymin').text)
                    xmax = int(box.find('xmax').text)
                    ymax = int(box.find('ymax').text)
                    coors = [xmin, ymin, xmax, ymax]
                    boxes.append(coors)

                return boxes, width, height

            def load_mask(self, image_id):
                """Load the mask for the given image ID."""
                info = self.image_info[image_id]
                path = info['annotation']
                boxes, w, h = self.extract_boxes(path)
                masks = np.zeros([h, w, len(boxes)], dtype='uint8')
                class_ids = []
                for i, box in enumerate(boxes):
                    masks[box[1]:box[3], box[0]:box[2], i] = 1
                    class_ids.append(1)  # Assume that there is only one class, 'tattoo'
                return masks.astype(np.bool), np.array(class_ids, dtype=np.int32)

            def image_reference(self, image_id):
                """Return the path of the image with the given ID."""
                info = self.image_info[image_id]
                return info['path']

        # Load the list of image filenames and create labels based on their names
        dataset_dir     = self.config['dataset_dir']
        images_dir      = os.path.join(dataset_dir, 'images')
        annotations_dir = os.path.join(dataset_dir, 'annots')
        image_filenames = [f for f in os.listdir(images_dir) if f.endswith(('.jpg'))]

        # Split the filenames into training and validation sets
        train_filenames, val_filenames = train_test_split(image_filenames, test_size=0.2, random_state=42)

        # Initialize and load the training dataset
        self.dataset_train = TattooDataset()
        self.dataset_train.load_dataset(images_dir, annotations_dir, train_filenames)
        self.dataset_train.prepare()

        # Initialize and load the validation dataset
        self.dataset_val = TattooDataset()
        self.dataset_val.load_dataset(images_dir, annotations_dir, val_filenames)
        self.dataset_val.prepare()

        print("Training and validation datasets generated.")

        # Visualize the datasets
        self.visualize_dataset(self.dataset_train, "Training")
        self.visualize_dataset(self.dataset_val, "Validation")

    def visualize_dataset(self, dataset, dataset_name, num_samples=2):
        """Visualize num_samples images from the dataset with their annotations."""
        print(f"{dataset_name} dataset:")
        print(f"Number of images: {len(dataset.image_ids)}")

        for _ in range(num_samples):
            image_id = random.choice(dataset.image_ids)
            image = dataset.load_image(image_id)
            
            # Se a imagem foi normalizada ou alterada, converta-a de volta para uint8
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            mask, class_ids = dataset.load_mask(image_id)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertendo de RGB para BGR se necessário

            for i in range(mask.shape[2]):
                box = np.where(mask[:, :, i] == 1)
                ymin, xmin = np.min(box, axis=1)
                ymax, xmax = np.max(box, axis=1)
                result_img = self.draw_transparent_rectangle(image_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 0.2)

            cv2.imshow(f"Image ID: {image_id} | {dataset_name}", result_img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
    # Helper function to draw a transparent rectangle on an image
    def draw_transparent_rectangle(self, img, top_left, bottom_right, color, alpha):
        """
        Desenha um retângulo preenchido com opacidade na imagem.

        Args:
        - img: Imagem original em que o retângulo será desenhado.
        - top_left: Tupla (x, y) do canto superior esquerdo do retângulo.
        - bottom_right: Tupla (x, y) do canto inferior direito do retângulo.
        - color: Cor do retângulo (B, G, R).
        - alpha: Opacidade do retângulo.

        Returns:
        - A imagem com o retângulo desenhado.
        """
        
        # Desenha um retângulo com borda mais grossa e escura sobre a imagem original
        cv2.rectangle(img, top_left, bottom_right, color, thickness=4)

        # Cria uma cópia da imagem para o preenchimento transparente
        overlay = img.copy()

        # Desenha um retângulo preenchido na cópia da imagem para o preenchimento transparente
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)

        output = img.copy()
        # Mistura a imagem original com a cópia modificada para adicionar o preenchimento transparente
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, output)

        return output

    def generate_model(self):
        """Generate the model."""
        print("Tattoo Localization!!", "Generate model...")

        train_size = len(self.dataset_train.image_ids)
        valid_size = len(self.dataset_val.image_ids)
        self.model = create_tattoo_loc_model(self.config, train_size, valid_size)

        # Load the pre-trained weights from COCO and exclude the output layers
        self.model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        print("Model generated and weights loaded.")

    def train_model(self):
        """Train the model."""
        print("Tattoo Localization!!", "Training...")

        # Create a callback that saves the model with the lowest validation loss
        checkpoint_path = "./models/tattoo/model_{epoch:02d}_{val_loss:.2f}.h5"
        # checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

        self.model.train(
            train_dataset=self.dataset_train,
            val_dataset=self.dataset_val,
            learning_rate=self.config.get("learning_rate", 0.001),
            epochs=self.config.get("epochs", 10),
            layers='heads'
        )

    def export_model(self):
        """Export the model."""
        print("Tattoo Localization!!", "Exporting data...")
        pass

    def locate(self):
        """Locate the tattoos in the images."""
        print("Tattoo Localization!!", "Detecting...")
        pass