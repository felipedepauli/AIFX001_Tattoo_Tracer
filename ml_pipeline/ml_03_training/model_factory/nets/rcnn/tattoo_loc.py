from os import listdir
from numpy import zeros
from numpy import asarray
from numpy import mean
from numpy import expand_dims
from xml.etree import ElementTree
from mrcnn.config import Config
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import extract_bboxes
from mrcnn.utils import compute_ap
from mrcnn.visualize import display_instances
import tensorflow as tf



from mrcnn.config import Config


# Inicializar a configuração
class BasicConfig(Config):
    # Dê um nome para a configuração. É útil se você estiver salvando pesos do modelo.
    NAME = "basic_cfg"
    
    # Número de classes (inclui a classe de fundo)
    NUM_CLASSES = 1 + 1  # Fundo + sua classe de interesse
    
    # Número de imagens para treinar em cada etapa
    IMAGES_PER_GPU = 1
    
    # Passos por época
    STEPS_PER_EPOCH = 10
    
    # Número de GPUs a serem utilizadas. Para CPU, manter como 1.
    GPU_COUNT = 1

    # Configurações adicionais podem ser adicionadas conforme necessário


def create_tattoo_loc_model(config, train_size, valid_size):
    
    class TattooLocalizationConfig(Config):
        NAME             = "tattoo_localization"
        NUM_CLASSES      = 1 + 1  # Background + tattoo
        GPU_COUNT        = 1
        IMAGES_PER_GPU   = config.get("batch_size", 4)
        STEPS_PER_EPOCH  = train_size // IMAGES_PER_GPU if train_size % IMAGES_PER_GPU == 0 else (train_size // IMAGES_PER_GPU) + 1
        VALIDATION_STEPS = valid_size // IMAGES_PER_GPU if valid_size % IMAGES_PER_GPU == 0 else (valid_size // IMAGES_PER_GPU) + 1
        LEARNING_RATE    = config.get("learning_rate", 0.001)
        #     #LEARNING_MOMENTUM = 0.9

#     #RPN_NMS_THRESHOLD = 0.7
        BACKBONE_STRIDES = [4, 8, 16, 32, 64, 96, 128]
        RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512, 768, 1024)
#     #RPN_ANCHOR_RATIOS = [0.5, 1, 2]
        WEIGHT_DECAY = 0.3
        TRAIN_BN = True
        ROI_POSITIVE_RATIO = 0.7
        TRAIN_ROIS_PER_IMAGE = 512
        DETECTION_NMS_THRESHOLD = 0.7
        IMAGE_MIN_DIM = 128
        MAX_GT_INSTANCES = 120
        DETECTION_MAX_INSTANCES = 120
    
    # Instanciar a configuração básica
    final_config = TattooLocalizationConfig()
    # final_config = TattooLocalizationConfig()
    model = MaskRCNN(mode='training', model_dir='./', config=final_config)
    return model


    
    
#     return model



# def extract_boxes(filename):
#     # load and parse the file
#     tree = ElementTree.parse(filename)
#     # get the root of the document
#     root = tree.getroot()
#     # extract each bounding box
#     boxes = list()
#     for box in root.findall('.//bndbox'):
#         xmin = int(box.find('xmin').text)
#         ymin = int(box.find('ymin').text)
#         xmax = int(box.find('xmax').text)
#         ymax = int(box.find('ymax').text)
#         coors = [xmin, ymin, xmax, ymax]
#         boxes.append(coors)
#     # extract image dimensions
#     width = int(root.find('.//size/width').text)
#     height = int(root.find('.//size/height').text)
#     return boxes, width, height

# class TattooDataset(Dataset):
#     # load the dataset definitions
#     def load_dataset(self, dataset_dir, is_train=True):
#         # define one class
#         self.add_class("dataset", 1, "tattoo")
#         # define data locations
#         images_dir = dataset_dir + '/images/'
#         annotations_dir = dataset_dir + '/annots/'
#         # find all images
#         for filename in listdir(images_dir):
#             # extract image id
#             image_id = filename[:-4]
            
#             if(image_id.endswith(' (1)')):
#               continue
              
#             # skip all images after 150 if we are building the train set
#             if is_train and int(image_id) > 3257:
#                 continue
#             # skip all images before 150 if we are building the test/val set
#             if not is_train and int(image_id) <= 3257:
#                 continue
#             img_path = images_dir + filename
#             ann_path = annotations_dir + image_id + '.xml'
#             # add to dataset
#             self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

#     # extract bounding boxes from an annotation file
#     def extract_boxes(filename):
#         # load and parse the file
#         tree = ElementTree.parse(filename)
#         # get the root of the document
#         root = tree.getroot()
#         # extract each bounding box
#         boxes = list()
#         for box in root.findall('.//bndbox'):
#             xmin = int(box.find('xmin').text)
#             ymin = int(box.find('ymin').text)
#             xmax = int(box.find('xmax').text)
#             ymax = int(box.find('ymax').text)
#             coors = [xmin, ymin, xmax, ymax]
#             boxes.append(coors)
#         # extract image dimensions
#         width = int(root.find('.//size/width').text)
#         height = int(root.find('.//size/height').text)
#         return boxes, width, height

#     # load the masks for an image
#     def load_mask(self, image_id):
#         # get details of image
#         info = self.image_info[image_id]
#         # define box file location
#         path = info['annotation']
#         # load XML
#         boxes, w, h = extract_boxes(path)
#         #boxes, w, h = self.extract_boxes(path)
#         # create one array for all masks, each on a different channel
#         masks = zeros([h, w, len(boxes)], dtype='uint8')
#         # create masks
#         class_ids = list()
#         for i in range(len(boxes)):
#             box = boxes[i]
#             row_s, row_e = box[1], box[3]
#             col_s, col_e = box[0], box[2]
#             masks[row_s:row_e, col_s:col_e, i] = 1
#             class_ids.append(self.class_names.index('tattoo'))
#         return masks, asarray(class_ids, dtype='int32')

#     # load an image reference
#     def image_reference(self, image_id):
#         info = self.image_info[image_id]
#         return info['path']



# # # test/val set
# # test_set = TattooDataset()
# # test_set.load_dataset('tattoo', is_train=False)
# # test_set.prepare()
# # print('Test: %d' % len(test_set.image_ids))

# class TattooConfig(Config):
#     #LEARNING_MOMENTUM = 0.9
#     #LEARNING_RATE = 0.001
    
#     # define the name of the configuration
#     NAME = "tattoo_cfg"
#     # number of classes (background + tattoo)
#     NUM_CLASSES = 1 + 1
#     # number of training steps per epoch
#     STEPS_PER_EPOCH = len(train_set.image_ids) // 2
#     #STEPS_PER_EPOCH = 1350
    
#     #RPN_NMS_THRESHOLD = 0.7
#     VALIDATION_STEPS = 407
#     BACKBONE_STRIDES = [4, 8, 16, 32, 64, 96, 128]
#     RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512, 768, 1024)
#     #RPN_ANCHOR_RATIOS = [0.5, 1, 2]
#     WEIGHT_DECAY = 0.5
#     TRAIN_BN = True
#     ROI_POSITIVE_RATIO = 0.7
#     TRAIN_ROIS_PER_IMAGE = 512
#     DETECTION_NMS_THRESHOLD = 0.7
#     IMAGE_MIN_DIM = 128
#     MAX_GT_INSTANCES = 120
#     DETECTION_MAX_INSTANCES = 120

# # prepare config
# config = TattooConfig()
# config.display()
# # define the model
# model = MaskRCNN(mode='training', model_dir='./', config=config)
# # load weights (mscoco) and exclude the output layers
# model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# with tf.device('/device:GPU:0'):
#     # train weights (output layers or 'heads')
#     model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=50, layers='heads')
  
# history = model.keras_model.history.history
# print()
# print()
# print('RELATORIO FINAL DO TREINAMENTO')
# print()
# print()
# print(history)

# with open('history050_FT15.txt', 'w') as f:
#   f.write(str(history))