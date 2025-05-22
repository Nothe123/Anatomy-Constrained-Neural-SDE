from sqlite3 import Timestamp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import SimpleITK as sitk 
import pandas as pd
import ants
sys.path.append('C:\\Users\\Administrator\\source\\repos\\NSDE\\NSDE') 
from OrganIntersectionConstraint import OrganIntersectionConstraint
from SmoothnessConstraint import SmoothnessConstraint
from BladderVolumeConstraint import BladderVolumeConstraint
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import time
import argparse



def read_pca_fields(mean_field_path, principal_components_folder):
    
    mean_field_tensor = read_mhd_image(mean_field_path)
    #mean_field_tensor = torch.tensor(mean_field, dtype=torch.float32)

    principal_components = []
    file_list = os.listdir(principal_components_folder)
    num = 0;
    for file_name in file_list:
        if file_name.startswith("component_") and file_name.endswith(".mhd"):
            file_path = os.path.join(principal_components_folder, file_name)
            component = read_mhd_image(file_path)
            principal_components.append(component)
            if(num==10):
                break
            
    principal_components_tensor = torch.tensor(np.array(principal_components), dtype=torch.float32)
    return mean_field_tensor, principal_components_tensor


def read_mhd_image(file_path):   
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    return image_tensor

def save_tensor_as_mhd(tensor, file_path, reference_image=None):
    # Convert Tensor to numpy array
    np_array = tensor.cpu().numpy()
    
    # If Tensor has channel dimension, select the first channel
    #if len(np_array.shape) == 4:
    #    np_array = np_array[0]  # Select first channel
    
    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(np_array)
    
    # Copy metadata from reference image if provided
    if reference_image is not None:
        sitk_image.CopyInformation(reference_image)
    
    # Save as MHD file
    sitk.WriteImage(sitk_image, file_path)
    print(f"Tensor saved to {file_path}")


class SDEModel(torchsde.SDEIto):
    def __init__(self, input_size,ref_blader, ref_rectum, ref_prostate, mean,vectors):
        super(SDEModel, self).__init__(noise_type="diagonal")
        self.drift=nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            #nn.Linear(32, 32),
            #nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, input_size) 
        )
        #self.drift = nn.Sequential(
        #    nn.Linear(input_size, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, input_size)
        #)
        self.diffusion = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            #nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, input_size) 
        )
        self.organ_intersection_constraints = OrganIntersectionConstraint(ref_blader, ref_rectum, ref_prostate, mean, vectors,input_size)                                                                      
        self.bladder_volume_constraints = BladderVolumeConstraint(ref_blader, mean, vectors,input_size, 0.1)
        
        
    def construct_orthonormal_basis(self,v):
        e1 = v / torch.linalg.norm(v)
        k = v.shape[0]
        random_vectors = torch.randn(k, k-1, device=v.device, dtype=v.dtype)
        full_matrix = torch.cat([e1.unsqueeze(1), random_vectors], dim=1)

        Q, _ = torch.linalg.qr(full_matrix)
        return Q
    def calculate_basis_matrix(self, pca_coeffs):
        try:
            
            
            self.organ_intersection_constraints.SetPCACoefficients(pca_coeffs)
            self.bladder_volume_constraints.SetPCACoefficients(pca_coeffs)
            
            B1 = self.organ_intersection_constraints.run()
            B2 = self.bladder_volume_constraints.run().unsqueeze(1)
            
            
            vect = torch.bmm(B2, B1).squeeze(1)
            vect = F.normalize(vect, p=2, dim=-1)  
            batch_size, dim = vect.shape
            extended = torch.randn(batch_size, dim, dim, device=vect.device)
            extended[:, :, 0] = vect
            basis_matrix= torch.linalg.qr(extended, mode='complete').Q 
            finalbasis_matrix = torch.bmm(B1, basis_matrix)
            del B1, B2, vect, extended, basis_matrix
            torch.cuda.empty_cache() 
               
            return finalbasis_matrix
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    def f(self, t, y):
        #print(f"t value: {t if isinstance(t, float) else t.item()}")
        hidden =self.drift(y)
        biasvector = torch.zeros_like(hidden)
        biasvector[:,0]=1
        hidden_biased= hidden+biasvector
        BaseVectors = self.calculate_basis_matrix(y)
        hidden_biased = hidden_biased.unsqueeze(1)
        output = torch.bmm(hidden_biased,BaseVectors)
        self.DriftOutput = output.squeeze(1)
        

        return self.DriftOutput

    def g(self, t, y):


        hidden =self.diffusion(y)
        biasvector = torch.zeros_like(hidden)
        biasvector[:,0]=1
        hidden_biased= hidden+biasvector
        BaseVectors = self.calculate_basis_matrix(y)
        hidden_biased = hidden_biased.unsqueeze(1)
        output = torch.bmm(hidden_biased,BaseVectors)
        self.DiffusionOutput = 0.3*output.squeeze(1)
        return self.DiffusionOutput




def train_model(model, input_size, train_loader,val_loader, refimg,mean, vector,num_epochs=1000, dt=1, lr=0.01, device='cpu',modelpath='bestmodel.pth'):
    model.to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    smoothness_constraint = SmoothnessConstraint(refimg, mean, vector,input_size)
    lamda=0.1
    scaler = GradScaler()

    drift_params = list(model.drift.parameters())
    diffusion_params = list(model.diffusion.parameters())
    best_val_loss=torch.tensor(float('inf'))
    patience=5
    for epoch in range(num_epochs):
        for param in drift_params:
            param.requires_grad = True
        for param in diffusion_params:
            param.requires_grad = True
            
        model.train()
        total_loss = 0
        for m_data in train_loader:
            m_data=m_data[0].to(device)
            m_data_shape=m_data.shape
            pca_coeffs = m_data[:,:,:m_data_shape[2]-1]
            timestamps = m_data[0,:,m_data_shape[2]-1]
           


            #pca_coeffs = pca_coeffs[0].to(device)
            #timestamps = mtimestamps.to(device)

            #input_size = pca_coeffs.shape[-1]
            t = torch.linspace(0, (pca_coeffs.size(1) - 1) * dt, pca_coeffs.size(1)).to(device)
            #print("Generated t:", t) 
            t[1:]+=5
            y0 = pca_coeffs[:, 0]
            if y0.dim() != 2:
                y0 = y0.unsqueeze(0)
            #with torch.autograd.profiler.profile(use_cuda=True) as prof:
            
            with autocast():
                start_time = time.time()
                ys = torchsde.sdeint(model, y0, timestamps, method="euler",dt=dt)
            #ys = torchsde.sdeint(model, y0, t,method="srk")
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"time {elapsed_time:.6f}")
            
            
                if ys.shape != pca_coeffs.shape:

                    ys = ys.permute(1, 0, 2)
            
                mle_loss = -torch.distributions.Normal(ys.squeeze(1), 1e-3).log_prob(pca_coeffs).sum()
                dy_dt_all = (ys[:, 1:] - ys[:, :-1]) / dt
                smoothness_constraint.SetPCAcoefficients(ys[:, :-1])
                smoothness_loss = smoothness_constraint.run(dy_dt_all)


                loss = mle_loss + lamda * smoothness_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            for param in drift_params:
                param.requires_grad = False
            for param in diffusion_params:
                param.requires_grad = False
            
            model.eval()

            total_val_loss = 0
            
            for m_data in val_loader:
                m_data = m_data[0].to(device)
                m_data_shape = m_data.shape
                pca_coeffs = m_data[:, :, :m_data_shape[2]-1]
                timestamps = m_data[0, :, m_data_shape[2]-1]

                t = torch.linspace(0, (pca_coeffs.size(1) - 1) * dt, pca_coeffs.size(1)).to(device)
                t[1:] += 5
                y0 = pca_coeffs[:, 0]
                if y0.dim() != 2:
                     y0 = y0.unsqueeze(0)
                with autocast():
                    ys = torchsde.sdeint(model, y0, timestamps, method="euler", dt=dt)

                    if ys.shape != pca_coeffs.shape:
                        ys = ys.permute(1, 0, 2)

                    mle_loss = -torch.distributions.Normal(ys.squeeze(1), 1e-3).log_prob(pca_coeffs).sum()
                    dy_dt_all = (ys[:, 1:] - ys[:, :-1]) / dt
                    smoothness_constraint.SetPCAcoefficients(ys[:, :-1])
                    smoothness_loss = smoothness_constraint.run(dy_dt_all)

                    loss = mle_loss + lamda * smoothness_loss

                total_val_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss}")
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), modelpath)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break
    return model



def read_pca_data(folder_path,timestamp_path=None):
    data = []
    timestamps=[]
    file_list = os.listdir(folder_path)
    index= 0
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        pca_coeffs = np.load(file_path)
        datalength = pca_coeffs.shape[0]
        if timestamp_path!=None:

            timestamp_filepath = timestamp_path+file_name.replace('.npy','_timestamp.npy')
            
             
            timestamp =np.load(timestamp_filepath)
           

            m_timestamp=np.expand_dims(timestamp[:datalength],axis=1)
            m_data = np.concatenate((pca_coeffs,m_timestamp),axis=1)
            
            new=m_data[:4,:]
                      

            data.append(new)
        else:
            data.append(pca_coeffs)

        

    return data
def generate_3d_dvf(pca_coeffs,vectors,mean):
        batch_size = pca_coeffs.shape[0]
        num_components = pca_coeffs.shape[1] 
        a_expanded = pca_coeffs.view(batch_size, num_components, 1, 1, 1, 1)
        b_expanded = vectors.unsqueeze(0)  
        component_dvf = a_expanded * b_expanded
        dvf = component_dvf.sum(dim=1)
        del a_expanded, b_expanded, component_dvf
        torch.cuda.empty_cache() if dvf.is_cuda else None

        mean_expanded = mean.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        dvf += mean_expanded
        return dvf

def generate_dvf_by_points_new(transform_files, reference_image):
    shape = reference_image.shape
    origin = np.array(reference_image.origin)
    spacing = np.array(reference_image.spacing)
    dimension = reference_image.dimension
    
    displacement_array = np.zeros(shape + (dimension,), dtype=np.float32)
    
    grid = np.stack(np.mgrid[[slice(0, s) for s in shape]], axis=-1)
    all_indices = grid.reshape(-1, dimension)
    
    physical_points = origin + all_indices * spacing
    
    if dimension == 2:
        columns = ['x', 'y']
    elif dimension == 3:
        columns = ['x', 'y', 'z']
    else:
        columns = [f'x{i}' for i in range(dimension)]  
    
    point_df = pd.DataFrame(physical_points, columns=columns)
    
    print("Applying transforms to all points...")
    transformed_df = ants.apply_transforms_to_points(
        dim=dimension,
        points=point_df,
        transformlist=transform_files
    )
    transformed_points = transformed_df.to_numpy()
    #np.array([
    #    transformed_df[columns].values for col in columns
    #]).T
    
    
    displacements = transformed_points - physical_points
    oldDVF=ants.image_read(transform_files[1])
    
    print("Updating displacement array...")
    displacement_array[tuple(all_indices.T)] = displacements

    newDVF=ants.from_numpy(
        displacement_array,
        origin=oldDVF.origin,
        spacing=oldDVF.spacing,
        direction=oldDVF.direction,
        has_components=oldDVF.has_components  
        )

    
    
    return newDVF

def predict(model, data_loader, refimg, mean, vector, dt=0.5, device='cpu',ref_path = None,Output_path=None,Ref_DVFPath =None,Simulation_times=100,test_path=None):
    model.to(device)
    
    ref_img = sitk.ReadImage(ref_path)

    all_params = list(model.parameters())
    for param in all_params:
            param.requires_grad = True
    model.eval()
   

    
    
    drift_params = list(model.drift.parameters())
    diffusion_params = list(model.diffusion.parameters())
    for param in drift_params:
        param.requires_grad = False
    #for param in diffusion_params:
    #    param.requires_grad = False

    
    predictions = []
    timestamps = torch.arange(10, 1010, 10).to(device)
    testindex=0

    
    index = 1
    
    for m_data in data_loader:
        

        m_data=m_data[0].to(device)
        m_data_shape=m_data.shape
        pca_coeffs = m_data[:,:,:m_data_shape[2]-1]
        timestamps = m_data[0,:,m_data_shape[2]-1]

        testimage_path = test_path+"/testimage"+str(index)+".mhd"
        Ref_DVFPath = test_path+"/test_ref_dvf"+str(index)+".mhd"
        test_img=ants.image_read(testimage_path)
        index=index+1
            
        y0 = pca_coeffs[:, 0]
        if y0.dim() != 2:
            y0 = y0.unsqueeze(0)

        for k in range(Simulation_times):
            with autocast():
                ys = torchsde.sdeint(model, y0, timestamps, method="euler", dt=dt)   
                if ys.shape != (len(timestamps), 1, y0.shape[1]):
                    ys = ys.permute(1, 0, 2)
                    
                pca_coeffs_np = ys.detach().cpu().numpy()[0]
               
                    
                for t, timestamp in enumerate(timestamps.cpu().numpy()):
                    current_pca = pca_coeffs_np[t]
                    if current_pca.ndim == 1:
                        current_pca = current_pca.reshape(1, -1)
                    current_pca = torch.tensor(current_pca, device=device)
                        
       
                    dvf=generate_3d_dvf(current_pca,vector,mean)
                    dvf=dvf.squeeze(0)
                    dvf_path = Output_path+"/Test"+str(testindex)+"_dvf_"+str(t)+"_SimulationTime"+str(k)+".mhd"
                    save_tensor_as_mhd(dvf,dvf_path,ref_img)
                    filelist=[]
        
                    filelist.append(Ref_DVFPath)
                    filelist.append(dvf_path)
                    composed_transform=generate_dvf_by_points_new(filelist, test_img)

                    compose_output_path = Output_path+"/Final_Test"+str(testindex)+"_dvf_"+str(t)+"_SimulationTime"+str(k)+".mhd"
       
                    ants.image_write(composed_transform, compose_output_path)



def main():
    print("Please select operation mode:")
    print("1. Train model")
    print("2. Apply model")
    
    try:
        choice = int(input("Enter number (1-2): "))
    except ValueError:
        print("Error: Please enter a valid number")
        sys.exit(1)
    
   
    DataRootDir="D://Prostate/1816967/Series1"
    model_path = DataRootDir+"/best_model.pth"
    
    
    ref_blader_path = DataRootDir+"/R_bladder.mhd"
    ref_rectum_path = DataRootDir+"/R_rectum.mhd"
    ref_prostate_path = DataRootDir+"/R_CTV.mhd"
    ref_img_path = DataRootDir+"/R_Img.mhd"
    
    principal_components_path = DataRootDir+"/iPCAOutput/"
    trainingfolder_path = principal_components_path+"TrainingData"
    mean_field_path = principal_components_path+"mean.mhd"
    trainingtimestamp_path = principal_components_path+"Time/"

    vadiationdata_path =principal_components_path+"ValidationData"
    validationtimestamp_path = principal_components_path+"ValidationTime/"
   
    testdata_path = principal_components_path+"TestData"
    testtimestamp_path = principal_components_path+"TestTime/"
    testimg_path= DataRootDir+"/TestImage"

    output_path = DataRootDir+"/TestOutput"


    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_blader = read_mhd_image(ref_blader_path).to(device)
    ref_rectum = read_mhd_image(ref_rectum_path).to(device)
    ref_prostate = read_mhd_image(ref_prostate_path).to(device)
    ref_img = read_mhd_image(ref_img_path).to(device)
    mean_field, principal_components = read_pca_fields(mean_field_path, principal_components_path)
    mean_field = mean_field.to(device)
    principal_components = principal_components.to(device)
    
    trainingdata = read_pca_data(trainingfolder_path, trainingtimestamp_path)
    trainingdata = [torch.tensor(pca, dtype=torch.float32) for pca in trainingdata]

    validationdata = read_pca_data(vadiationdata_path,validationtimestamp_path)
    validationdata = [torch.tensor(pca, dtype=torch.float32) for pca in validationdata]

    testdata = read_pca_data(testdata_path,testtimestamp_path)
    testdata = [torch.tensor(pca, dtype=torch.float32) for pca in testdata]

    trainingdataset= TensorDataset(torch.stack(trainingdata))
    validationdataset = TensorDataset(torch.stack(validationdata))
    testdataset = TensorDataset(torch.stack(testdata))

    batch_size = 1
    train_loader = DataLoader(trainingdataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validationdataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)
    input_size =trainingdata[0].shape[-1]-1

    model = SDEModel(input_size,ref_blader,ref_rectum,ref_prostate,mean_field,principal_components)
    if choice == 1:

        trained_model = train_model(model,input_size, train_loader,validation_loader, ref_img,mean_field,principal_components,device=device,modelpath=model_path)
       
        print("Model training completed")
            
    elif choice == 2:
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        
        if model:
            print(f"Model loaded from {model_path}")
            predict(model=model, data_loader=test_loader, refimg=ref_img, mean=mean_field, vector=principal_components, device=device,ref_path = ref_img_path,Output_path=output_path,Ref_DVFPath =None,Simulation_times=100,test_path=testimg_path)
            
            print("Prediction completed")
    else:
        print("Error: Invalid choice. Valid choices are 1 or 2.")

if __name__ == "__main__":
    main()



