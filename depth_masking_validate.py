import json
import time

import numpy
import matplotlib
import matplotlib.pyplot as plt
import imageio

from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_segmentation_masks

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf 


import datasets
import networks

from utils.kitti_utils import *
from utils.train_utils import *
from utils.loss_utils import *
from utils.validation_utils import *


class DepthMaskingValidate:
    def __init__(self, options):
        self.opt = options
        timestr = time.strftime("%Y%m%d-%H%M%S")

        self.log_path = os.path.join(self.opt.val_log_dir, timestr)
        
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models = {}

        self.models["depth_encoder"] = networks.DepthEncoder()
        self.models["depth_encoder"].to(self.device)

        self.models["depth_decoder"] = networks.DepthDecoder(num_ch_enc=self.models["depth_encoder"].num_ch_enc)
        self.models["depth_decoder"].to(self.device)

        self.models["pose_mask_encoder"] = networks.PoseMaskEncoder()
        self.models["pose_mask_encoder"].to(self.device)
                
        self.models["pose_decoder"] = networks.PoseDecoder(self.models["pose_mask_encoder"].num_ch_enc)
        self.models["pose_decoder"].to(self.device)

        self.models["mask_decoder"] = networks.MaskDecoder(self.models["pose_mask_encoder"].num_ch_enc, 
                                                           num_output_channels=self.opt.num_K)
        self.models["mask_decoder"].to(self.device)
        
        
        # Used to load models
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Model named:\n  ", self.opt.model_name)
        print("Device using:\n  ", self.device)

        # datasets
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'


        val_dataset = self.dataset(self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                                   self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers,
                                     pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        
        # Define depth backprojection and 3D projection helper functions 
        self.backproject_depth = BackprojectDepth(self.opt.batch_size, self.opt.height, self.opt.width)
        self.backproject_depth.to(self.device)

        self.project_3d = Project3D(self.opt.batch_size, self.opt.height, self.opt.width, device=self.device)
        self.project_3d.to(self.device)
        
        # Tensorboard data logger
        self.writers = {}
        for mode in ["val"]:
            self.writers[mode] = SummaryWriter(log_dir=os.path.join(self.log_path))
        self.step = 0
        
        print("Using split:\n  ", self.opt.split)
        print("There are validation items\n".format(len(val_dataset)))
    
    

    def run_validation(self):
        
        self.load_model()
        
        for i in range(self.opt.num_val_batches): 
            self.val()
            self.step += 1
    
    
    def set_eval(self):
        """Convert all networks to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
            
            
    def val(self):
        """Validate the model on a single minibatch
        """ 
        self.set_eval()
        
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs = self.process_batch(inputs)
            self.log("val", inputs, outputs)
            
            del inputs, outputs

    
    def process_batch(self, inputs):
            """Pass a minibatch through the network and generate images and losses
            """
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            features = self.models["depth_encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth_decoder"](features)

            disp = outputs[("disp", 0)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                
            outputs[("depth", 0, 0)] = depth
            
            pose_mask_inputs = torch.cat([outputs["depth", 0, 0], 
                                        inputs[("color_aug", -1, 0)], 
                                        inputs[("color_aug", 0, 0)]], 1)
            
            pose_mask_features = self.models["pose_mask_encoder"](pose_mask_inputs)

                
            masks = self.models["mask_decoder"](pose_mask_features)

            # masks.shape: (batch, K, height, width)
            masks = masks[('masks',0)]
            
            # cam_points_masks.shape: (batch, K, height*width)
            cam_points_masks = masks.reshape(self.opt.batch_size, self.opt.num_K, -1)
            
            # axisangle.shape, translation.shape: (batch, K, 1, 3)
            axisangle, translation = self.models["pose_decoder"](pose_mask_features)
            
            transformations = []     
            for k in range(self.opt.num_K):
                # key: param; frame; scale; K
                outputs[("axisangle", 0, 0, k)] = axisangle[:, k]
                outputs[("translation", 0, 0, k)] = translation[:, k]
                
                # Invert the matrix if the frame id is negative
                transformation = transformation_from_parameters(axisangle[:, k], translation[:, k], invert=True)
                
                # transfomation.shape:(batch, 4, 4)
                outputs[("cam_T_cam", 0, 0, k)] = transformation
                transformations.append(transformation)
                
            # cam_points.shape: (batch, 4, height*width)
            cam_points = self.backproject_depth(depth, inputs[("inv_K", 0)])
            
            # pix_coords.shape: (batch, height, width, 2)
            pix_coords = self.project_3d(cam_points, inputs[("K", 0)], transformations, cam_points_masks)
            
            outputs[("sample", 0, 0)] = pix_coords
            
            outputs[("color", 0, 0)] = F.grid_sample(inputs[("color", 0, 0)], 
                                                    outputs[("sample", 0, 0)], 
                                                    padding_mode="border")
            
            outputs[("color_masked", 0, 0)] = inputs[("color_aug", 0, 0)]
            segmentation_colors = ['red', 'green', 'blue', 'orange', 'black']
          
            heatmap = plt.get_cmap('plasma')
            outputs[("heatmap", 0, 0)] = np.zeros((self.opt.batch_size, self.opt.height, self.opt.width, 3), dtype=np.uint8)
            
            for i in range(self.opt.batch_size):
                image = inputs[("color_aug", 0, 0)][i]
                image_int = tf.convert_image_dtype(image, dtype=torch.uint8)
                 
                mask_labels = masks[i].argmax(0) == torch.arange(masks.shape[1], device=self.device)[:, None, None]
                image_with_all_masks = draw_segmentation_masks(image_int.to('cpu'), 
                                                               masks=mask_labels.to('cpu'), 
                                                               alpha=0.2, 
                                                               colors=segmentation_colors)
                outputs[("color_masked", 0, 0)][i] = image_with_all_masks

                depth_image = outputs['depth', 0, 0][i].data.cpu().numpy().transpose((1, 2, 0))
                
                heatmap_image_rgba = heatmap(depth_image)
                heatmap_image_rgba = Image.fromarray(np.uint8(heatmap_image_rgba*255), 'RGBA')
                
                heatmap_image_rgb = heatmap_image_rgba.convert('RGB')
                heatmap_image = np.asarray(heatmap_image_rgb)

                outputs[("heatmap", 0, 0)][i] = np.asarray(heatmap_image, dtype=np.uint8)
            
            
            return outputs
    
    
    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)    

    
    def log(self, mode, inputs, outputs):
        """Write an event to the tensorboard events file
        """
        print("Logging outputs on batch: {}".format(self.step))
        
        writer = self.writers[mode]
               
        for j in range(self.opt.batch_size):
            writer.add_image("color_{}_{}/{}".format(-1, 0, j), 
                            inputs[("color", -1, 0)][j].data, 
                            self.step)
            
            writer.add_image("color_{}_{}/{}".format(0, 0, j),
                            inputs[("color", 0, 0)][j].data, 
                            self.step)
            
            writer.add_image("color_pred_{}_{}/{}".format(0, 0, j),
                            outputs[("color", 0, 0)][j].data, 
                            self.step)
            
            writer.add_image("depth{}_{}/{}".format(0, 0, j),
                            outputs[("depth", 0, 0)][j].data, 
                            self.step)
            
            writer.add_image("depth_heatmap{}_{}/{}".format(0, 0, j),
                            outputs[("heatmap", 0, 0)][j], 
                            self.step, dataformats='HWC')
            
            writer.add_image("color_masked{}_{}/{}".format(0, 0, j),
                            outputs[("color_masked", 0, 0)][j].data, 
                            self.step)
            
            writer.add_image("disp_{}/{}".format(0, j), 
                             normalize_image(outputs[("disp", 0)][j]), 
                             self.step)
