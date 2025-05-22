import SimpleITK as sitk
import numpy as np
from sklearn.decomposition import IncrementalPCA
from pathlib import Path
from tqdm import tqdm
import joblib
import os
def load_and_preprocess(filepath, downsample_factor=4):
    
    
    image = sitk.ReadImage(str(filepath))
    dvf_array = sitk.GetArrayFromImage(image) 
    
    
    if downsample_factor > 1:
        dvf_array_new = dvf_array[::downsample_factor, 
                             ::downsample_factor, 
                             ::downsample_factor, :]
    else:
        dvf_array_new=dvf_array


    
    return dvf_array_new.flatten()

def save_ipca_model(ipca, output_dir, original_shape):
    Path(output_dir).mkdir(exist_ok=True)
    
    
    components_reshaped = ipca.components_.reshape(-1, *original_shape)
    for i, component in enumerate(components_reshaped):
        component_image = sitk.GetImageFromArray(component)
        sitk.WriteImage(component_image, f"{output_dir}/component_{i}.mhd")
    

    mean_reshaped = ipca.mean_.reshape(original_shape)
    mean_image = sitk.GetImageFromArray(mean_reshaped)
    sitk.WriteImage(mean_image, f"{output_dir}/mean.mhd")
    
    np.save(f"{output_dir}/explained_variance.npy", ipca.explained_variance_)
    joblib.dump(ipca, f"{output_dir}/full_model.pkl")

def save_coefficients(transformed_data, output_filename):
    np.save(output_filename, transformed_data)

def calculate_reconstruction_error(ipca, file_paths, batch_size,downsample_factor):
    
    total_mse = 0.0
    n_samples = len(file_paths)
    
    for i in tqdm(range(0, n_samples, batch_size), desc="Calculating Reconstruction Error"):
        batch_files = file_paths[i:i+batch_size]
        X_batch = np.array([load_and_preprocess(f,downsample_factor) for f in batch_files])
        
        
        X_trans = ipca.transform(X_batch)
        X_recon = ipca.inverse_transform(X_trans)
        
        
        batch_mse = np.mean((X_batch - X_recon) ** 2, axis=1)
        total_mse += np.sum(batch_mse)
    
    return total_mse / n_samples
def load_and_use_model(model_dir):
    ipca = joblib.load(f"{model_dir}/full_model.pkl")
    
    new_data = load_and_preprocess("new_sample.mhd")
    new_coeff = ipca.transform([new_data])
    reconstructed = ipca.inverse_transform(new_coeff)
    
    return new_coeff, reconstructed

def apply_model(model_dir,test_dir,test_pat_num,outputdir):
    ipca = joblib.load(f"{model_dir}/full_model.pkl")
    transformed_data = []
    for i in range(test_pat_num):
        test_pat_path = test_dir+"/test"+str(i)
        transformed_data = []
        idx=0
        for filename in os.listdir(test_pat_path):
            file_path = os.path.join(test_dir, filename)
            new_data = load_and_preprocess(file_path)
            new_coeff = ipca.transform([new_data])
            transformed_data = np.vstack(new_coeff)
            outputfilename = outputdir+"pat"+str(i)+"_fr"+str(idx)+".npy"
    
            save_coefficients(transformed_data, outputfilename)
            idx=idx+1
   

def main(data_dir, n_components=50, batch_size=5, downsample_factor=4):
    
    file_paths = list(Path(data_dir).glob("*.mhd"))
    if not file_paths:
        raise FileNotFoundError("No mhd file exists")
    sample = load_and_preprocess(file_paths[0], downsample_factor)
    ipca = IncrementalPCA(n_components=n_components, 
                         batch_size=batch_size)
    print(f"iPCAstarts(Target dimension: {n_components})")
    for i in tqdm(range(0, len(file_paths), batch_size), desc="training progress"):
        batch_files = file_paths[i:i+batch_size]
        X_batch = np.array([load_and_preprocess(f, downsample_factor) 
                           for f in batch_files])
        ipca.partial_fit(X_batch)
    avg_mse = calculate_reconstruction_error(ipca, file_paths, batch_size,downsample_factor)
    print("\nResult")
    print(f"explained_variance_ratio_: {np.sum(ipca.explained_variance_ratio_):.3f}")
    print(f"avg_MSE: {avg_mse:.6f}")
    print(f"FinalDimension: {n_components} PC")

    output_dir='D://Prostate/NewData/iPCAOutput'

    transformed_data = []
    for i in tqdm(range(0, len(file_paths), batch_size), desc="Generated iPCA Coefficients"):
        batch_files = file_paths[i:i+batch_size]
        X_batch = np.array([load_and_preprocess(f, downsample_factor) 
                           for f in batch_files])
        transformed_data.append(ipca.transform(X_batch))
    
    transformed_data = np.vstack(transformed_data)
    
    image = sitk.ReadImage(str(file_paths[0]))
    dvf_array = sitk.GetArrayFromImage(image) 

    OutShape = list(dvf_array.shape)
    OutShape[0]=int(dvf_array.shape[0]/downsample_factor)
    OutShape[1]=int(dvf_array.shape[1]/downsample_factor)
    OutShape[2]=int(dvf_array.shape[2]/downsample_factor)
    
   
    save_ipca_model(ipca, output_dir, OutShape)
    save_coefficients(transformed_data, output_dir)

    image = sitk.ReadImage(str(file_paths[0]))
    dvf_array = sitk.GetArrayFromImage(image) 
    
    
    with open(f"{output_dir}/metadata.txt", "w") as f:
        f.write(f"original_shape: {dvf_array.shape}\n")
        f.write(f"n_components: {n_components}\n")
        f.write(f"downsample_factor: {downsample_factor}\n")
    test_dir="D://Prostate/NewData/GeneratedDVF/testdvf"
    outputdir="D://Prostate/NewData/GeneratedDVF/testdata"
    apply_model(output_dir,test_dir,10,outputdir)


if __name__ == "__main__":
    CONFIG = {
        "data_dir": "D://Prostate/NewData/GeneratedDVF",  
        "n_components": 10,        
        "batch_size": 10,           
        "downsample_factor":1     
    }
    
    main(**CONFIG)