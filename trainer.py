import json

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import datasets
import networks

from utils.kitti_utils import *
from utils.train_utils import *
from utils.loss_utils import *
from utils.validation_utils import *


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the Depth Decoder/Encoders and add parameters to training list
        self.models["depth_encoder"] = networks.DepthEncoder()
        self.models["depth_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["depth_encoder"].parameters())

        self.models["depth_decoder"] = networks.DepthDecoder(num_ch_enc=self.models["depth_encoder"].num_ch_enc)
        self.models["depth_decoder"].to(self.device)
        self.parameters_to_train += list(self.models["depth_decoder"].parameters())

        self.models["pose_mask_encoder"] = networks.PoseMaskEncoder()
        self.models["pose_mask_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_mask_encoder"].parameters())
                
        self.models["pose_decoder"] = networks.PoseDecoder(self.models["pose_mask_encoder"].num_ch_enc)
        self.models["pose_decoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_decoder"].parameters())

        self.models["mask_decoder"] = networks.MaskDecoder(self.models["pose_mask_encoder"].num_ch_enc, 
                                                           num_output_channels=self.opt.num_K)
        self.models["mask_decoder"].to(self.device)
        self.parameters_to_train += list(self.models["mask_decoder"].parameters())
        
        
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        # Used to load models during training; you can specify which models to update weights for in options
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # datasets
        # Note: Use raw dataset split or odom split (different sequences of data?)
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        


        # Initialize the training and validation dataloaders
        train_dataset = self.dataset(self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                                     self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers,
                                       pin_memory=True, drop_last=True)

        val_dataset = self.dataset(self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                                   self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers,
                                     pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.ssim = SSIM()
        self.ssim.to(self.device)
        
        # Define depth backprojection and 3D projection helper functions 
        self.backproject_depth = BackprojectDepth(self.opt.batch_size, self.opt.height, self.opt.width)
        self.backproject_depth.to(self.device)

        self.project_3d = Project3D(self.opt.batch_size, self.opt.height, self.opt.width, device=self.device)
        self.project_3d.to(self.device)
        
        # Tensorboard data loggers
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(log_dir=os.path.join(self.log_path, mode))

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))

        self.save_opts()


    def set_train(self):
        """Convert all networks to training mode
        """
        for m in self.models.values():
            m.train()


    def set_eval(self):
        """Convert all networks to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()


    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()


    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            # Added
            if batch_idx == 0:
                self.model_lr_scheduler.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    compute_depth_losses(inputs, outputs, losses, self.depth_metric_names)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
            

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
        
        # Make mask/pose encoder input by concatenating: predicted target depth, source image, and target image
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
                
        losses = self.compute_losses(inputs, outputs)
                
        return outputs, losses


    def compute_losses(self, inputs, outputs, smooth_coefficient=0.001, alpha=0.85):
        """Compute the photo and smooth losses for a minibatch
        """
        color_target = inputs["color_aug", 0, 0]
        color_pred = outputs[("color", 0, 0)]
        depth_pred = outputs[("depth", 0, 0)]

        photo_loss = torch.mean((1 - alpha) * torch.abs(color_target - color_pred) + (alpha/2) * (1 - self.ssim.forward(color_target, color_pred)))
        smooth_loss = get_smooth_loss(depth_pred, color_target)

        losses = {}
        losses["loss"] = photo_loss + smooth_coefficient * smooth_loss
        losses["photo"] = photo_loss
        losses["smooth"] = smooth_loss

        return losses
    
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
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                compute_depth_losses(inputs, outputs, losses, self.depth_metric_names)

            self.log("val", inputs, outputs, losses)
            
            del inputs, outputs, losses

        self.set_train()



    # MODEL AND HYPERPARAMETER OPTIONS UTILITY FUNCTIONS
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "networks")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)


    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "networks", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)


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

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
            

    def log_time(self, batch_idx, duration, loss):
            """Print a logging statement to the terminal
            """
            
            samples_per_sec = self.opt.batch_size / duration
            time_sofar = time.time() - self.start_time
            training_time_left = (
                self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} | time elapsed: {} | time left: {}"
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))


    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        
        writer = self.writers[mode]
        
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.step)):  # write a maxmimum of four images
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
            
            writer.add_image("disp_{}/{}".format(0, j), 
                             normalize_image(outputs[("disp", 0)][j]), 
                             self.step)
