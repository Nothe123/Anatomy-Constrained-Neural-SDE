import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import SimpleITK as sitk
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import os
#from skimage.transform import resize



class MRIImageDataset(Dataset):
    def __init__(self, data_dir,  transform=None):
        """
        Initialize medical image dataset without using glob
        
        Args:
            data_dir: Path to data directory containing sparse_volumetric, reference_3d, and sorted_cine subdirectories
            split: 'train' or 'val' for training/validation split
            transform: Data augmentation transforms
        """
        self.transform = transform
        self.data_dir = data_dir
       
        
        
        self.sparse_dir = os.path.join(data_dir, 'sparse_volumetric')
        self.reference_dir = os.path.join(data_dir, 'reference_3d')
        self.cine_dir = os.path.join(data_dir, 'label')
        
        
        try:
            self.case_ids = [
                filename.split('.')[0] 
                for filename in os.listdir(self.sparse_dir) 
                if filename.endswith('.mhd')
            ]
        except FileNotFoundError:
            self.case_ids = []
        
        
        random.seed(42)
        random.shuffle(self.case_ids)
        

    def __len__(self):
        """Return dataset size"""
        return len(self.case_ids)

    def __getitem__(self, idx):
        """Get a training data pair"""
        case_id = self.case_ids[idx]
        
        # Construct file paths
        sparse_path = os.path.join(self.sparse_dir, f'{case_id}.mhd')
        reference_path = os.path.join(self.reference_dir, f'{case_id}.mhd')
        cine_path = os.path.join(self.cine_dir, f'{case_id}.mhd')
        
        # Load medical images
        sparse_volumetric = self._load_image(sparse_path)
        reference_3d = self._load_image(reference_path)
        sorted_cine = self._load_image(cine_path)
        
        # Apply data augmentation
        if self.transform and self.split == 'train':
            sparse_volumetric, reference_3d, sorted_cine = self.transform(
                sparse_volumetric, reference_3d, sorted_cine
            )
        
        return sparse_volumetric, reference_3d, sorted_cine

    def _load_image(self, path):
       
        img = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(img)
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)  
        return img_tensor
        
    



class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x
class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        self.block11 = ConvBlock3D(1, 4)
        self.block12 = ConvBlock3D(4, 4)
        self.bn1 = nn.BatchNorm3d(4)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block21 = ConvBlock3D(4, 8)
        self.block22 = ConvBlock3D(8, 8)
        self.bn2 = nn.BatchNorm3d(8)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)


        self.block31 = ConvBlock3D(8, 16)
        self.block32 = ConvBlock3D(16, 16)
        self.bn3 = nn.BatchNorm3d(16)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 30 * 50 * 50, 1024)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(1024, 64)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.block12(self.block11(x)))))
        x = self.pool2(self.relu2(self.bn2(self.block22(self.block21(x)))))
        x = self.pool3(self.relu3(self.bn3(self.block32(self.block31(x)))))

        ss=x.size()
        
        x = x.view(-1, 16 * 30 * 50 * 50)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class ReferenceEncoder(nn.Module):
    def __init__(self):
        super(ReferenceEncoder, self).__init__()
        self.block11 = ConvBlock3D(1, 4)
        self.block12 = ConvBlock3D(4, 4)
        self.bn1 = nn.BatchNorm3d(4)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block21 = ConvBlock3D(4, 8)
        self.block22 = ConvBlock3D(8, 8)
        self.bn2 = nn.BatchNorm3d(8)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)


        self.block31 = ConvBlock3D(8, 16)
        self.block32 = ConvBlock3D(16,16)
        self.bn3 = nn.BatchNorm3d(16)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 30 * 50 * 50, 1024)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(1024, 64)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.block12(self.block11(x)))))
        x = self.pool2(self.relu2(self.bn2(self.block22(self.block21(x)))))
        x = self.pool3(self.relu3(self.bn3(self.block32(self.block31(x)))))
        
        x = x.view(-1, 16 * 30 * 50 * 50)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class DVFReconstructionDecoder(nn.Module):
    def __init__(self):
        super(DVFReconstructionDecoder, self).__init__()
        self.fc1 = nn.Linear(128, 32 * 30 * 50 * 50)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)

        

        self.deconv2 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block21 = ConvBlock3D(16, 16)
        self.block22 = ConvBlock3D(16, 16)
        self.bn2 = nn.BatchNorm3d(16)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)

        self.deconv3 = nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block31 = ConvBlock3D(8, 8)
        self.block32 = ConvBlock3D(8, 8)
        self.bn3 = nn.BatchNorm3d(8)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)

        self.deconv4 = nn.ConvTranspose3d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block41 = ConvBlock3D(3, 3)
        self.block42 = ConvBlock3D(3, 3)
        self.bn4 = nn.BatchNorm3d(3)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = x.view(-1, 32, 30, 50, 50)
        
        x = self.relu3(self.bn2(self.block22(self.block21(self.deconv2(x)))))
        x = self.relu4(self.bn3(self.block32(self.block31(self.deconv3(x)))))
        x = self.relu5(self.bn4(self.block42(self.block41(self.deconv4(x)))))
       
        return x


class DVFEstimationNetwork(nn.Module):
    def __init__(self):
        super(DVFEstimationNetwork, self).__init__()
        self.motion_encoder = MotionEncoder()
        self.reference_encoder = ReferenceEncoder()
        self.decoder = DVFReconstructionDecoder()

    def forward(self, sparse_volumetric, reference_3d):
        motion_features = self.motion_encoder(sparse_volumetric)
        reference_features = self.reference_encoder(reference_3d)
        combined_features = torch.cat([motion_features, reference_features], dim=1)
        dvf = self.decoder(combined_features)
        return dvf

def wrap(reference_image, dvf):
    batch_siz, channel,depth, height, width = reference_image.size()
    
   
    z = torch.linspace(-1, 1, depth, device=reference_image.device)
    y = torch.linspace(-1, 1, height, device=reference_image.device)
    x = torch.linspace(-1, 1, width, device=reference_image.device)
    
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0)
    
    
    norm_dvf = torch.zeros_like(dvf)
    norm_dvf[:, 0, :, :, :] = 2 * dvf[:, 0, :, :, :] / (width - 1)
    norm_dvf[:, 1, :, :, :] = 2 * dvf[:, 1, :, :, :] / (height - 1)
    norm_dvf[:, 2, :, :, :] = 2 * dvf[:, 2, :, :, :] / (depth - 1)
    
    
    warped_grid = grid + norm_dvf
    warped_grid = warped_grid.permute(0, 2, 3, 4, 1)
    
    
    warped_image = F.grid_sample(
        reference_image, 
        warped_grid, 
        mode='bilinear', 
        padding_mode='border',
        align_corners=True
    )
    
    
    del grid, norm_dvf, warped_grid
    torch.cuda.empty_cache()
    
    return warped_image


def mi_loss(x, y, bins=32, sigma=0.5):
    
    x = x.float()
    y = y.float()
    
    
    x_min, x_max = torch.min(x), torch.max(x)
    y_min, y_max = torch.min(y), torch.max(y)
    
    x = (x - x_min) / (x_max - x_min + 1e-8)
    y = (y - y_min) / (y_max - y_min + 1e-8)
    
    
    batch_size, channels, depth, height, width = x.size()
    x = x.view(batch_size, -1)
    y = y.view(batch_size, -1)
    
    
    x_indices = torch.clamp(torch.floor(x * bins).long(), 0, bins - 1)
    y_indices = torch.clamp(torch.floor(y * bins).long(), 0, bins - 1)
    
    
    hist_2d = torch.zeros(batch_size, bins, bins, device=x.device)
    for b in range(batch_size):
        
        indices = x_indices[b] * bins + y_indices[b]
        hist = torch.bincount(indices, minlength=bins*bins).reshape(bins, bins)
        hist_2d[b] = hist
    
   
    kernel_size = int(2 * 3 * sigma + 1)  # 3-sigma rule
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure odd size
    
    
    x = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
    x = x - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.reshape(1, 1, kernel_size, kernel_size)
    
    
    hist_2d_smooth = F.conv2d(
        hist_2d.unsqueeze(1),  
        kernel_2d,
        padding=kernel_size // 2
    ).squeeze(1)  
    
    
    p_x = torch.sum(hist_2d_smooth, dim=2) + 1e-8
    p_y = torch.sum(hist_2d_smooth, dim=1) + 1e-8
    
    p_xy = hist_2d_smooth / torch.sum(hist_2d_smooth, dim=[1, 2], keepdim=True)
    p_x = p_x / torch.sum(p_x, dim=1, keepdim=True)
    p_y = p_y / torch.sum(p_y, dim=1, keepdim=True)
    
    
    mask_xy = p_xy > 0
    mask_x = p_x > 0
    mask_y = p_y > 0
    
    h_xy = -torch.sum(p_xy[mask_xy] * torch.log(p_xy[mask_xy]))
    h_x = -torch.sum(p_x[mask_x] * torch.log(p_x[mask_x]))
    h_y = -torch.sum(p_y[mask_y] * torch.log(p_y[mask_y]))
    
    
    mi = h_x + h_y - h_xy
    return -mi / batch_size


def relative_mse_loss(y_pred, y_true, epsilon=1e-8):
    return torch.mean(torch.square(y_pred - y_true) / (torch.square(y_true) + epsilon))

def total_loss(reference_mri, label, dvf, lambda_=10,stage=1):
    reconstructed_mri = wrap(reference_mri,dvf)
    if stage==1:
        
        sim_loss = mi_loss(reconstructed_mri, label)
    else:
        sim_loss = mi_loss(reconstructed_mri, label)
    dvf_loss = dvf_regularization_loss(dvf)
    return sim_loss + lambda_ * dvf_loss


def dvf_regularization_loss(dvf):
    loss = 0
    for m in range(3):
       
        component = dvf[:, m, :, :, :]
        
        
        d_dvf_dx = torch.gradient(component, dim=1)[0]  # Depth dimension
        d_dvf_dy = torch.gradient(component, dim=2)[0]  # Height dimension
        d_dvf_dz = torch.gradient(component, dim=3)[0]  # Width dimension
        
        
        loss += torch.sum(torch.square(d_dvf_dx))
        loss += torch.sum(torch.square(d_dvf_dy))
        loss += torch.sum(torch.square(d_dvf_dz))
    
    return loss



def save_tensor_as_mhd(tensor, output_path):
   
    numpy_array = tensor.cpu().numpy()

    
    if numpy_array.shape[0] == 1:
        numpy_array = numpy_array.squeeze(0)

   
    sitk_image = sitk.GetImageFromArray(numpy_array)

  
    sitk.WriteImage(sitk_image, output_path)

def apply_model(model, testdataloader, device,outputpath):
    model.eval()
    with torch.no_grad():
        for i, (sparse_volumetric, reference_3d, label) in enumerate(testdataloader):
            sparse_volumetric = sparse_volumetric.to(device)
            reference_3d = reference_3d.to(device)

            dvf = model(sparse_volumetric, reference_3d)
            reconstructed_mri = wrap(reference_3d, dvf)
            OutDVFname = outputpath+'\\DVF'+str(i)+".mhd"
            save_tensor_as_mhd(dvf,OutDVFname)
            OutImgname = outputpath+'\\ReconImg'+str(i)+".mhd"
            save_tensor_as_mhd(reconstructed_mri,OutImgname)
            
            del sparse_volumetric, reference_3d, label, dvf, reconstructed_mri
            torch.cuda.empty_cache()
    return 0

def train_model(model, train_dataloader, val_dataloader, num_epochs, optimizer, stage, device, save_dir='./models', patience=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    



    if stage == 2:
        stage1_model_path = f'{save_dir}/model_stage1_best.pth'
        if os.path.exists(stage1_model_path):
            print(f"Loading stage 1 model from {stage1_model_path}")
            
            model.load_state_dict(torch.load(stage1_model_path, map_location=device)['model_state_dict'])
            
           
            for param in model.parameters():
                param.requires_grad = False
                
            for param in model.motion_encoder.block21.parameters():
                param.requires_grad = True

            for param in model.motion_encoder.block22.parameters():
                param.requires_grad = True
                
            for param in model.motion_encoder.fc1.parameters():
                param.requires_grad = True
            for param in model.motion_encoder.fc2.parameters():
                param.requires_grad = True
                
            print("Stage 1 model loaded successfully. Training specific layers only.")
        else:
            raise FileNotFoundError(f"Stage 1 model not found at {stage1_model_path}. Cannot proceed with stage 2.")

    
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_epoch = 0
    
    model.train()
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        model.train()
        for i, (sparse_volumetric, reference_3d, label) in enumerate(train_dataloader):
            sparse_volumetric = sparse_volumetric.to(device)
            reference_3d = reference_3d.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            dvf = model(sparse_volumetric, reference_3d)
            reconstructed_mri = wrap(reference_3d, dvf)
            
            

            loss = total_loss(reference_3d, label, dvf, stage=stage)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            del sparse_volumetric, reference_3d, label, dvf, reconstructed_mri
            torch.cuda.empty_cache()

        train_loss = running_train_loss / len(train_dataloader)
        
        running_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, (sparse_volumetric, reference_3d, label) in enumerate(val_dataloader):
                sparse_volumetric = sparse_volumetric.to(device)
                reference_3d = reference_3d.to(device)
                label = label.to(device)

                dvf = model(sparse_volumetric, reference_3d)
                reconstructed_mri = wrap(reference_3d, dvf)
                
                
                loss = total_loss(reference_3d, label, dvf, stage=stage)

                running_val_loss += loss.item()
                del sparse_volumetric, reference_3d, label, dvf, reconstructed_mri
                torch.cuda.empty_cache()

        val_loss = running_val_loss / len(val_dataloader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Stage {stage}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        diff = (val_loss-best_val_loss)/best_val_loss
        
        if diff<-0.05:
            best_val_loss = val_loss
            best_model_epoch = epoch + 1
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'stage': stage
            }, f'{save_dir}/model_stage{stage}_best.pth')
        else:
            early_stop_counter += 1
            print(f'Early stopping counter: {early_stop_counter}/{patience}')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'stage': stage
        }, f'{save_dir}/model_stage{stage}_epoch_{epoch+1}.pth')
        
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            print(f'Best model saved at epoch {best_model_epoch} with val loss: {best_val_loss:.4f}')
            break
    
    return best_val_loss, best_model_epoch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DVFEstimationNetwork().to(device)
    
    while True:
        print("\nSelect operation:")
        print("1. Train model")
        print("2. Apply model")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ")
        
        if choice == '1':
            stage = int(input("Select training stage (1/2): "))
            if stage not in [1, 2]:
                print("Invalid stage, please select 1 or 2")
                continue
                
            

            trainingdatadir = 'D:\\ReconProstateData\TrainingData'
            validationdatadir = 'D:\\ReconProstateData\ValidationData'
            
            train_dataset = MRIImageDataset(trainingdatadir)
            val_dataset = MRIImageDataset(validationdatadir)
            
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            
            optimizer = optim.Adam(
                [p for p in model.parameters() if p.requires_grad], 
                lr=0.001
            )
            
            num_epochs =10
            
            
            best_val_loss, best_epoch = train_model(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=num_epochs,
                optimizer=optimizer,
                stage=stage,
                device=device,
                save_dir='D:\\ReconProstateModel',
                patience=0.1
            )
            
            print(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            
        elif choice == '2':
            testdatadir = 'D:\\ReconProstateData\TestData'
            
            test_dataset = MRIImageDataset(testdatadir)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            model_path = 'D:\\ReconProstateModel\model_stage2_best.pth'
            model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
            
            result = apply_model(model, test_dataloader, device,outputpath='D:\\ReconOutput')
            print("Reconstruction complete, result shape:", result.shape)
            
        elif choice == '3':
            break
        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    main()  
