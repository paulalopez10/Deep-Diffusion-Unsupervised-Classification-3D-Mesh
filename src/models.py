import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch.nn import Module
import torch_geometric.transforms as T
from utils import retrieval
from torch.autograd import Variable
import numpy as np
import random


def pw_cosine_distance(input_a, input_b):
   normalized_input_a = torch.nn.functional.normalize(input_a)  
   normalized_input_b = torch.nn.functional.normalize(input_b)
   res = torch.mm(normalized_input_a, normalized_input_b.T)
   res *= -1 # 1-res without copy
   res += 1
   return res


def ksparse(x, k):
    values, indices = torch.topk(x, k=k, dim=-1)
    values = values[..., -1]
    values = values.reshape(-1, 1)
    y = torch.where(x < values, torch.zeros_like(x), x)
    return y

class STN3d(torch.nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 9)
        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(torch.nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k*k)
        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(torch.nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x#, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1)#, trans, trans_feat



class CustomPointCloudDataAugmentation:
    """Module to perform custom data augmentation on point cloud data."""

    def __init__(self, rotation_degrees=(-5, 5), scaling_range=(0.8, 1.2), shear_factor=0.2, jitter_range=0.2):
        super().__init__()

        self.rotation_degrees = rotation_degrees
        self.scaling_range = scaling_range
        self.shear_factor = shear_factor
        self.jitter_range = jitter_range

    def __call__(self, point_cloud):
        # Apply random rotation
        angle_x = torch.tensor(random.uniform(self.rotation_degrees[0], self.rotation_degrees[1]))
        angle_y = torch.tensor(random.uniform(self.rotation_degrees[0], self.rotation_degrees[1]))
        angle_z = torch.tensor(random.uniform(self.rotation_degrees[0], self.rotation_degrees[1]))
        point_cloud = self.rotate_point_cloud(point_cloud, angle_x, angle_y, angle_z)

        # Apply random scaling
        self.scale_factor = random.uniform(self.scaling_range[0], self.scaling_range[1])
        point_cloud = self.scale_point_cloud(point_cloud)

        # Apply random shearing
        point_cloud = self.shear_point_cloud(point_cloud)

        # Apply random jitter (translation)
        translation = torch.rand(3) * 2 * self.jitter_range - self.jitter_range
        point_cloud += translation.cuda()

        return point_cloud

    def rotate_point_cloud(self, point_cloud, angle_x, angle_y, angle_z):
        cos_theta_x = torch.cos(torch.deg2rad(angle_x))
        sin_theta_x = torch.sin(torch.deg2rad(angle_x))
        rotation_matrix_x = torch.tensor([[1, 0, 0],
                                            [0, cos_theta_x, -sin_theta_x],
                                            [0, sin_theta_x, cos_theta_x]])
        cos_theta_y = torch.cos(torch.deg2rad(angle_y))
        sin_theta_y = torch.sin(torch.deg2rad(angle_y))
        rotation_matrix_y = torch.tensor([[cos_theta_y, 0, sin_theta_y],
                                            [0, 1, 0],
                                            [-sin_theta_y, 0, cos_theta_y]])
        cos_theta_z = torch.cos(torch.deg2rad(angle_z))
        sin_theta_z = torch.sin(torch.deg2rad(angle_z))
        rotation_matrix_z = torch.tensor([[cos_theta_z, -sin_theta_z, 0],
                                            [sin_theta_z, cos_theta_z, 0],
                                            [0, 0, 1]])
        rotated_point_cloud = torch.matmul(point_cloud, rotation_matrix_x.cuda())
        rotated_point_cloud = torch.matmul(rotated_point_cloud, rotation_matrix_y.cuda())
        rotated_point_cloud = torch.matmul(rotated_point_cloud, rotation_matrix_z.cuda())
        return rotated_point_cloud

    def scale_point_cloud(self, point_cloud):
        scaled_point_cloud = point_cloud * self.scale_factor
        return scaled_point_cloud

    def shear_point_cloud(self, point_cloud):
        sheared_point_cloud = point_cloud + self.shear_factor * point_cloud[:, 1].unsqueeze(1)
        return sheared_point_cloud


class Net_DeepDiff(pl.LightningModule):
    def __init__(self, num_sources, hidden_features, num_outputs,  sub_model, n_knn, param_lambda, batch_size, x_sources):
        super().__init__()
        self.x_sources = x_sources
        self.batch_size = batch_size
        self.param_lambda = param_lambda
        self.num_outputs = num_outputs
        self.num_sources = num_sources        
        self.x_diffusion_source = torch.rand(self.batch_size, self.num_sources)
        self.sub_model = Module()
        self.param_knn = n_knn
        self.transform =  CustomPointCloudDataAugmentation(
                                            rotation_degrees=(-5, 5),
                                            scaling_range=(0.8, 1.2),
                                            shear_factor=0.2,
                                            jitter_range=0.2
                                        )
        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.training_step_outputs = []
        self.training_step_labels = []
        
        
        
        self.sub_model = sub_model(global_feat = True, feature_transform = False)
        self.fc1 = Linear(hidden_features, num_outputs) # hidden_features must be 1024 for pointnet
        self.bn1 = BatchNorm1d(num_outputs)
        
        self.WeightMat =Linear( self.num_outputs, self.num_sources)
        torch.nn.init.xavier_normal_(self.WeightMat.weight, gain=2.0)
        
        self.predictions = []
        self.targets = []
        

    def forward(self, data):

        
        input_data = []
       
        for i in range(int(data['x'].batch[-1])+1):
            if torch.rand(1)>0.2 and self.trainer.training:
                input_data.append(torch.transpose(self.transform(data['x'][i].pos),0,1))
            else:       
                input_data.append(torch.transpose(data['x'][i].pos,0,1))
          
                
        x = self.sub_model(torch.stack(input_data))
     
        encoded = F.relu(self.bn1(self.fc1(x)))
        
        feat_normalized = F.normalize(encoded + 1e-8, p=2, dim=1) # + 1e-8
        if self.current_epoch==0 and self.training: #initialize M with random projection
             with torch.no_grad():
                 for i , j  in enumerate(data['idx']):
                     self.WeightMat.weight[self.x_sources[j.item()].bool()] = feat_normalized[i] #feat_normalized[i]
        
        
        y_predicted = self.WeightMat(feat_normalized)
        return  feat_normalized , y_predicted   #Normalize for feature retrieval
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def process_batch(self, batch):
        out, y_predicted = self.forward(batch)
        lab = batch['y'].squeeze()
        xx=[]
        for j in batch['idx']:
            xx.append(self.x_sources[j.item()])
        self.x_diffusion_source = torch.stack(xx) #extract IDs from batch and generate matrix

        loss = self.deepdiffusion_loss(out, y_predicted) 

        return loss, out, lab

    def training_step(self, batch, batch_idx):
        loss, pred, lab = self.process_batch(batch)
        self.training_step_outputs.append(pred)
        self.training_step_labels.append(lab)
        self.log('train_loss', loss[0], batch_size=len(batch['y']))
        self.log('train_loss_smooth', loss[1], batch_size=len(batch['y']))
        self.log('train_loss_fitting', loss[2], batch_size=len(batch['y']))
   
        return loss[0]

    def validation_step(self, batch, batch_idx):
        loss, pred, lab = self.process_batch(batch)
        self.validation_step_outputs.append(pred)
        self.validation_step_labels.append(lab)
        self.log('val_loss', loss[0], batch_size=len(batch['y']))
        self.log('val_loss_smooth', loss[1], batch_size=len(batch['y']))
        self.log('val_loss_fitting', loss[2], batch_size=len(batch['y']))
     

    def on_test_start(self):
        self.predictions = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        out, y = self.forward(batch)
        #prob = torch.softmax(out, dim=1) #TODO check
        self.predictions.append(out)
        self.targets.append(batch['y'].squeeze())

    def on_validation_epoch_end(self):
        all_preds = torch.vstack(self.validation_step_outputs)
        lab = torch.hstack(self.validation_step_labels)
        auc = retrieval(all_preds, lab)

        self.log('val_auc', auc, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory
        self.validation_step_labels.clear()

    def on_train_epoch_end(self):
        all_preds = torch.vstack(self.training_step_outputs)
        lab = torch.hstack(self.training_step_labels)
        auc = retrieval(all_preds, lab) 
        self.log('train_auc', auc, prog_bar=True)
        self.training_step_outputs.clear()  # free memory
        self.training_step_labels.clear()


    def deepdiffusion_loss(self, out, y_predicted):#, WeightMat, x_diffusion_source, param_knn, param_lambda):

        S = pw_cosine_distance(out,self.WeightMat.weight)
        W = ksparse(S,self.param_knn)
        

        mask = torch.not_equal(W, torch.zeros_like(W))
        nonzero_idx = torch.nonzero(mask)
        nonzero_val = torch.masked_select(W, mask).view([-1, 1]) #Wbn


        a = y_predicted[nonzero_idx[:, 0]]
        b = self.WeightMat.weight[nonzero_idx[:, 1]]
        a = torch.nn.functional.softmax(a, dim=1, dtype=torch.double) #Same as self.WeightMat(a)
        b = torch.nn.functional.softmax(self.WeightMat(b), dim=1, dtype=torch.double)

        # compute the Jensen–Shannon divergence
        a = torch.clamp(a, 1e-10, 1.0)
        b = torch.clamp(b, 1e-10, 1.0)

        a = torch.distributions.Categorical(probs=a)
        b = torch.distributions.Categorical(probs=b)
        kld_ab = torch.reshape(torch.distributions.kl_divergence(a, b), [-1, 1])
        kld_ba = torch.reshape(torch.distributions.kl_divergence(b, a), [-1, 1])
        jsd = kld_ab + kld_ba

        loss_smoothness = torch.sum(nonzero_val * jsd)


        loss_fitting = torch.sum(F.cross_entropy(y_predicted, self.x_diffusion_source.cuda()))

        loss =  self.param_lambda *loss_smoothness +  loss_fitting

        return loss , loss_smoothness, loss_fitting





