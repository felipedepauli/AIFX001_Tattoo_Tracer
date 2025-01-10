import os
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model
import cv2

class TattooDataset(mrcnn.utils.Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        # The first thing we have to do is create the class we want to predict
        # This method adds information (image ID, image path, and annotations) about each image into a dictionary
        self.add_class("dataset", 1, "tattoo")
        
        # Now we need the images and annotations. Then we get the image ID, and with it we build the path names
        images_dir = dataset_dir + '/images/'
        annots_dir = dataset_dir + '/annots/'
        count = 0
            
        for filename in os.listdir(images_dir):
            image_id = os.path.splitext(filename)[0]
            if is_train:
                if count > 4000:
                    break
                else:
                    count += 1
            elif not is_train:
                if count < 4000:
                    count += 1
                    continue
            
            img_path = images_dir + filename
            ann_path = annots_dir + image_id + '.xml'
            
            # Just add all together as an image
            self.add_image("dataset", image_id=image_id, path=img_path, annotation=ann_path)
            
    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        #boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('tattoo'))
        return masks, asarray(class_ids, dtype='int32')
        
    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)
        root = tree.getroot()
        
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height
        

train_dataset = TattooDataset()
train_dataset.load_dataset('./datasets/tattoo_trace/loc/', is_train=True)
train_dataset.prepare()

valid_dataset = TattooDataset()
valid_dataset.load_dataset('./datasets/tattoo_trace/loc/', is_train=False)
valid_dataset.prepare()


class TattooConfig(mrcnn.config.Config):
    NAME = "tattoo"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 131


tattoo_config = TattooConfig()

model = mrcnn.model.MaskRCNN(mode='training', model_dir='./logs', config=tattoo_config)

model.load_weights(filepath='mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset, valid_dataset, 0.0001, epochs=1, layers='heads')

history = model.keras_model.history.history
print()
print()
print('RELATORIO FINAL DO TREINAMENTO')
print()
print()
print(history)

with open('history050_FT15.txt', 'w') as f:
  f.write(str(history))
  
model_path = './modelo_tattoo.h5'
model.keras_model.save_weights(model_path)

print("Modelo salvo em:", model_path)