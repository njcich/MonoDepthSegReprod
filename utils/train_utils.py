import os
import hashlib
import zipfile
from six.moves import urllib

import numpy as np
import pdb 

import torch
import torch.nn as nn

import time


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)


    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, device, eps=1e-7):
        super(Project3D, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        
        
    def forward(self, points, K, transformations, cam_points_masks):
        '''
        points: torch.Size([12, 4, 122880])
        K.shape: torch.Size([12, 4, 4])
        T.shape: torch.Size([12, 4, 4])
        P.shape: torch.Size([12, 3, 4])
        cam_points.shape: torch.Size([12, 3, 122880]) - transformed points
        
        |_  before 
        
        new : 
        transfomations.shape: k length list torch.Size([batch, 4, 4])
        cam_points_masks: (batch, K, height*width)
        
        '''
        # P is K*T 
        # cam_points = Px
        
        batch_size = cam_points_masks.shape[0]
        num_K = cam_points_masks.shape[1]
        num_points = cam_points_masks.shape[2]
        
        all_weighted_points = []
        for k in range(num_K):
            T_k =  transformations[k]
            P_k = torch.matmul(K, T_k)[:, :3, :]
            
            
            # mask.shape: (batch, num_points)
            mask =  cam_points_masks[:, k, :]
            # this has the weighting for transfomation k at point p for all samples in batch
            # want to multiply this weighting by the points 3D points, 2nd dim of points :3
            # then we can multiply these weight_points multiplying by P_k the transformation at mask k
            
            points_weighted = torch.ones(points.shape, device=self.device, requires_grad=False)
            for i in range(3):
                points_weighted[:,i,:] = mask*points[:,i,:]
            
            # transformed_weighted_points.shape: torch.Size([12, 3, 122880])
            transformed_weighted_points = torch.matmul(P_k, points_weighted)
            all_weighted_points.append(transformed_weighted_points)
        

        transformed_points = torch.zeros(transformed_weighted_points.shape, device=self.device, requires_grad=False)
        for i in range(3):
            for k in range(num_K):
                transformed_points[:,i,:] =+ all_weighted_points[k][:,i,:]
            
            
                
        #This is the original 
        # cam_points = torch.matmul(P, points)
        # pdb.set_trace()
        
        
        # above works for num_K > 1
        
        pix_coords = transformed_points[:, :2, :] / (transformed_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        
        return pix_coords


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    
    return scaled_disp, depth


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    
    return lines



# LOGGING UTILITY FUNCITONS
def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)






# def log_time(log_params, batch_idx, duration, loss):
#         """Print a logging statement to the terminal
#         """
        
#         samples_per_sec = log_params['batch_size']/ duration
#         time_sofar = time.time() - log_params['start_time']
#         training_time_left = (
#             log_params['num_total_steps'] / log_params['step'] - 1.0) * time_sofar if log_params['step'] > 0 else 0
#         print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
#             " | loss: {:.5f} | time elapsed: {} | time left: {}"
#         print(print_string.format(log_params['epoch'], batch_idx, samples_per_sec, loss,
#                                   sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))


# def log(writers, log_params, mode, inputs, outputs, losses):
#     """Write an event to the tensorboard events file
#     """
#     writer = writers[mode]
    
#     for l, v in losses.items():
#         writer.add_scalar("{}".format(l), v, log_params['step'])

#     for j in range(min(4, log_params['step'])):  # write a maxmimum of four images
        
#         writer.add_image("color_{}_{}/{}".format(-1, 0, j), 
#                          inputs[("color", -1, 0)][j].data, 
#                          log_params['step'])
                    
#         writer.add_image("color_pred_{}_{}/{}".format(0, 0, j),
#                          outputs[("color", 0, 0)][j].data, 
#                          log_params['step'])

#         writer.add_image("disp_{}/{}".format(0, j), normalize_image(outputs[("disp", 0)][j]), log_params['step'])

