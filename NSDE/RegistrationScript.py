import os
import re
import ants
import SimpleITK as sitk
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import pandas as pd


def load_mri_image(image_path):
    """Load MRI image in MHD format"""
    try:
        img = ants.image_read(image_path)
        return img
    except Exception as e:
        print(f"Failed to load image {image_path}: {str(e)}")
        return None


def load_mask(mask_path):
    """Load mask image in MHD format"""
    try:
        mask = ants.image_read(mask_path)
        return mask
    except Exception as e:
        print(f"Failed to load mask {mask_path}: {str(e)}")
        return None


def calculate_volume(mask):
    """Calculate the volume of a mask in mm^3"""
    if mask is None:
        return 0
    voxel_volume = np.prod(mask.spacing)  # mm^3 per voxel
    return np.sum(mask.numpy() > 0) * voxel_volume


def perform_rigid_registration(fixed_image, moving_image):
    """Perform rigid registration (translation + rotation)"""
    rigid_registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform='Rigid',
        verbose=False
    )
    return rigid_registration


def perform_deformable_registration(fixed_image, moving_image, mask,initial_transform=None):
    """Perform deformable registration with optional initial transform"""
    if initial_transform:
        # Apply initial rigid transform
        moving_image = ants.apply_transforms(
            fixed=fixed_image,
            moving=moving_image,
            transformlist=initial_transform['fwdtransforms']
        )
    
    deformable_registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform="SyNAggro",             
        reg_iterations=(100, 50, 0),
        mask=mask,
        moving_mask = mask,
        mask_all_stages=True,
        syn_metric='meansquares',
        flow_sigma=0.5,
        grad_step=2
    )

    return deformable_registration


def compose_transforms(transforms, reference_image):
    """
    Compose multiple transforms into a single transform
    
    Args:
    transforms: List of transforms to compose
    reference_image: Reference image to define the output space
    """
    composite_transform = ants.compose_transforms(
        transforms=transforms,
        reference=reference_image
    )
    return composite_transform


def invert_transform(transform):
    """Invert a transform"""
    if isinstance(transform, str):
        # Load transform if it's a file path
        transform = ants.image_read(transform)
    
    # Invert the transform
    inverted_transform = ants.invert_transform(transform)
    return inverted_transform


def process_patient_folder(patient_folder, output_base_folder):
    """
    Process a single patient folder
    
    Args:
    patient_folder: Path to the patient folder
    output_base_folder: Path to the output base folder
    """
    patient_id = os.path.basename(patient_folder)
    print(f"Processing patient: {patient_id}")
    
    # Get all fraction folders
    fraction_folders = [f for f in os.listdir(patient_folder) 
                       if os.path.isdir(os.path.join(patient_folder, f))]
    
    # Collect all setup MRI and their bladder masks
    setup_mris = []
    bladder_volumes = []
    
    for fraction_folder in fraction_folders:
        fraction_path = os.path.join(patient_folder, fraction_folder)
        fraction_id = os.path.basename(fraction_path)
        
        # Find setup MRI image
        setup_mri_path = find_setup_mri(fraction_path, patient_id, fraction_id)
        if not setup_mri_path:
            print(f"    Warning: Setup MRI not found in {fraction_id}")
            continue
        
        # Find organ mask
        bladder_mask_path = find_mask(fraction_path, patient_id, fraction_id, "Bladder")
        if not bladder_mask_path:
            print(f"    Warning: Bladder mask not found in {fraction_id}")
            continue

        prostate_mask_path = find_mask(fraction_path, patient_id, fraction_id, "Prostate")
        if not prostate_mask_path:
            print(f"    Warning: Prostate mask not found in {fraction_id}")
            continue

        rectum_mask_path = find_mask(fraction_path, patient_id, fraction_id, "Rectum")
        if not rectum_mask_path:
            print(f"    Warning: Rectum mask not found in {fraction_id}")
            continue
        
        # Load setup MRI and bladder mask
        setup_mri = load_mri_image(setup_mri_path)
        bladder_mask = load_mask(bladder_mask_path)
        prostate_mask = load_mask(prostate_mask_path)
        rectum_mask = load_mask(rectum_mask_path)
        
        if setup_mri is None :
            continue
        
        # Calculate bladder volume
        volume = calculate_volume(bladder_mask)
        setup_mris.append((setup_mri_path, setup_mri, fraction_id))
        bladder_volumes.append((volume, setup_mri_path, fraction_id))
    
    # Step 2: Select reference setup MRI based on median bladder volume
    if not setup_mris:
        print(f"No valid setup MRIs found for patient {patient_id}")
        return
    
    # Sort by bladder volume and select median
    bladder_volumes.sort(key=lambda x: x[0])
    median_index = len(bladder_volumes) // 2
    median_volume, reference_path, reference_fraction_id = bladder_volumes[median_index]
    print(f"Selected reference setup MRI from fraction {reference_fraction_id} with bladder volume: {median_volume:.2f} mm^3")
    
    # Load reference setup MRI
    reference_setup_mri = load_mri_image(reference_path)
    if reference_setup_mri is None:
        print(f"Failed to load reference setup MRI: {reference_path}")
        return
    
    # Process each setup MRI to register to the reference
    for setup_mri_path, setup_mri, fraction_id in setup_mris:
        if setup_mri_path == reference_path:
            print(f"Skipping reference setup MRI from fraction {fraction_id}")
            continue
        
        print(f"Registering setup MRI from fraction {fraction_id} to reference")
        
        # Create output directory
        output_registration_folder = os.path.join(
            output_base_folder, patient_id, f"registration_to_fraction_{reference_fraction_id}")
        os.makedirs(output_registration_folder, exist_ok=True)
        
        # Step 1: Rigid registration
        rigid_output_prefix = os.path.join(
            output_registration_folder, 
            f"rigid_f{fraction_id}_to_f{reference_fraction_id}_")
        
        rigid_registration = perform_rigid_registration(
            fixed=reference_setup_mri,
            moving=setup_mri
        )
        
        # Save rigid transform
        rigid_transforms = rigid_registration['fwdtransforms']
        for i, transform in enumerate(rigid_transforms):
            transform_path = f"{rigid_output_prefix}transform_{i}.mhd"
            ants.image_write(transform, transform_path)
        
        # Step 2: Deformable registration with rigid initial transform
        deformable_output_prefix = os.path.join(
            output_registration_folder, 
            f"deformable_f{fraction_id}_to_f{reference_fraction_id}_")
        
        deformable_registration = perform_deformable_registration(
            fixed=reference_setup_mri,
            moving=setup_mri,
            initial_transform=rigid_registration
        )
        
        # Save deformable transform
        deformable_transforms = deformable_registration['fwdtransforms']
        for i, transform in enumerate(deformable_transforms):
            transform_path = f"{deformable_output_prefix}transform_{i}.mhd"
            ants.image_write(transform, transform_path)
        
        # Step 3: Compose rigid and deformable transforms into a single DVF
        compose_output_path = os.path.join(
            output_registration_folder, 
            f"composed_f{fraction_id}_to_f{reference_fraction_id}.mhd")
        
        # Collect all forward transforms (rigid followed by deformable)
        all_transforms = rigid_transforms + deformable_transforms
        
        # Compose transforms
        composed_transform = compose_transforms(all_transforms, reference_setup_mri)
        
        # Save composed transform
        ants.image_write(composed_transform, compose_output_path)
        print(f"Composed transform saved to: {compose_output_path}")
        
        # Process reconstructed MRIs for this fraction
        process_reconstructed_mris(
            patient_folder, patient_id, fraction_id, 
            reference_setup_mri, reference_fraction_id,
            composed_transform, output_base_folder
        )


def process_reconstructed_mris_old(patient_folder, patient_id, fraction_id, 
                               reference_setup_mri, reference_fraction_id,
                               setup_to_reference_transform, output_base_folder):
    """
    Process reconstructed MRIs for a given fraction and create new composite transforms
    
    Args:
    patient_folder: Path to the patient folder
    patient_id: Patient ID
    fraction_id: Fraction ID
    reference_setup_mri: Reference setup MRI image
    reference_fraction_id: Reference fraction ID
    setup_to_reference_transform: Transform from current setup to reference setup
    output_base_folder: Path to the output base folder
    """
    fraction_path = os.path.join(patient_folder, fraction_id)
    
    # Find all reconstructed MRI images for this fraction
    reconstructed_mris = find_reconstructed_mris(fraction_path, patient_id, fraction_id)
    if not reconstructed_mris:
        print(f"    No reconstructed MRIs found in fraction {fraction_id}")
        return
    
    # Find setup MRI for this fraction
    setup_mri_path = find_setup_mri(fraction_path, patient_id, fraction_id)
    if not setup_mri_path:
        print(f"    Setup MRI not found in fraction {fraction_id}")
        return
    
    setup_mri = load_mri_image(setup_mri_path)
    if setup_mri is None:
        return
    
    # Create output directory
    output_composite_folder = os.path.join(
        output_base_folder, patient_id, f"composite_transforms_f{fraction_id}")
    os.makedirs(output_composite_folder, exist_ok=True)
    
    # Invert the setup_to_reference transform
    inverted_setup_to_reference_transform = invert_transform(setup_to_reference_transform)
    
    # Process each reconstructed MRI
    for recon_mri_path in reconstructed_mris:
        recon_mri_name = os.path.basename(recon_mri_path)
        recon_id = re.search(r'Recon_{0}_{1}_(\d+)\.mhd'.format(patient_id, fraction_id), recon_mri_name).group(1)
        print(f"  Processing reconstructed MRI: {recon_mri_name}")
        
        # Load reconstructed MRI
        recon_mri = load_mri_image(recon_mri_path)
        if recon_mri is None:
            continue
        
        # Find the DVF from setup to reconstructed MRI (generated in step 1)
        dvf_filename = f"DVF_setup_to_{os.path.splitext(recon_mri_name)[0]}.mhd"
        dvf_path = os.path.join(
            output_base_folder, patient_id, fraction_id, dvf_filename)
        
        if not os.path.exists(dvf_path):
            print(f"    Warning: Setup-to-reconstructed DVF not found: {dvf_path}")
            continue
        
        # Load the setup-to-reconstructed MRI DVF
        setup_to_recon_dvf = load_mri_image(dvf_path)
        
        # Step 3: Compose the inverted setup-to-reference transform with setup-to-reconstructed transform
        composite_output_path = os.path.join(
            output_composite_folder, 
            f"composite_f{fraction_id}_recon{recon_id}_to_reference.mhd")
        
        # Compose transforms: (setup-to-reference)^-1 followed by setup-to-reconstructed
        composite_transform = compose_transforms(
            transforms=[setup_to_recon_dvf, inverted_setup_to_reference_transform],
            reference=reference_setup_mri
        )
        
        # Save composite transform
        ants.image_write(composite_transform, composite_output_path)
        print(f"    Composite transform saved to: {composite_output_path}")


def find_setup_mri(fraction_folder, patient_id, fraction_id):
    pattern = re.compile(rf'Positioning_{patient_id}_{fraction_id}\.mhd', re.IGNORECASE)
    for file in os.listdir(fraction_folder):
        if pattern.match(file):
            return os.path.join(fraction_folder, file)
    return None


def find_mask(fraction_folder, patient_id, fraction_id, organ_name):
    """Find mask image with specific naming pattern"""
    pattern = re.compile(rf'Positioning_{organ_name}_{patient_id}_{fraction_id}\.mhd', re.IGNORECASE)
    for file in os.listdir(fraction_folder):
        if pattern.match(file):
            return os.path.join(fraction_folder, file)
    return None


def find_reconstructed_mris(fraction_folder, patient_id, fraction_id):
    """Find reconstructed MRI images with specific naming pattern"""
    reconstructed_mris = []
    pattern = re.compile(rf'Recon_{patient_id}_{fraction_id}_\d+\.mhd', re.IGNORECASE)
    for file in os.listdir(fraction_folder):
        if pattern.match(file):
            reconstructed_mris.append(os.path.join(fraction_folder, file))
    return reconstructed_mris



def oldmain():
    """Main function"""
    # Set data paths
    data_folder = "D:\\ProstatePatientData"  # Root directory of patient data
    output_folder = "/path/to/output"  # Root directory for output results
    
    # Get all patient folders
    patient_folders = [f for f in os.listdir(data_folder) 
                      if os.path.isdir(os.path.join(data_folder, f))]
    
    # Process each patient
    for patient_folder in tqdm(patient_folders, desc="Processing patients"):
        patient_path = os.path.join(data_folder, patient_folder)
        process_patient_folder(patient_path, output_folder)
    
    print("All patient data processing completed!")

def load_mri_with_masks(mri_path):
    """
    Load 3D-MRI image and corresponding bladder, rectum, and prostate masks
    
    Args:
        mri_path (str): Path to the 3D-MRI image
    
    Returns:
        tuple: (mri_image, bladder_mask, rectum_mask, prostate_mask)
               Returns None for any missing masks
    """
    if not os.path.exists(mri_path):
        raise FileNotFoundError(f"MRI file not found: {mri_path}")
    
    mri_dir = os.path.dirname(mri_path)
    mri_filename = os.path.basename(mri_path)
    
    match = re.match(r'([a-zA-Z]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)', mri_filename)
    if not match:
        raise ValueError(f"Failed to parse MRI filename: {mri_filename}")
    
    prefix, patient_id, fraction_id = match.groups()
    
    organ_masks = {
        'bladder': f"{prefix}_Bladder_{patient_id}_{fraction_id}.mhd",
        'rectum': f"{prefix}_Rectum_{patient_id}_{fraction_id}.mhd",
        'prostate': f"{prefix}_Prostate_{patient_id}_{fraction_id}.mhd"
    }
    
    try:
        mri_image = ants.image_read(mri_path)
    except Exception as e:
        print(f"Failed to load MRI image: {str(e)}")
        mri_image = None
    
    masks = {}
    for organ, mask_filename in organ_masks.items():
        mask_path = os.path.join(mri_dir, mask_filename)
        try:
            if os.path.exists(mask_path):
                masks[organ] = ants.image_read(mask_path)
                #print(f"Successfully loaded {organ} mask: {mask_path}")
            else:
                masks[organ] = None
                print(f"{organ} mask not found: {mask_path}")
        except Exception as e:
            masks[organ] = None
            print(f"Failed to load {organ} mask: {str(e)}")
    
    return mri_image, masks['bladder'], masks['rectum'], masks['prostate']

def load_reconmri_with_masks(mri_path):
    """
    Load 3D-MRI image and corresponding bladder, rectum, and prostate masks
    
    Args:
        mri_path (str): Path to the 3D-MRI image
    
    Returns:
        tuple: (mri_image, bladder_mask, rectum_mask, prostate_mask)
               Returns None for any missing masks
    """
    if not os.path.exists(mri_path):
        raise FileNotFoundError(f"MRI file not found: {mri_path}")
    
    mri_dir = os.path.dirname(mri_path)
    mri_filename = os.path.basename(mri_path)
    
    match = re.match(r'([a-zA-Z]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)', mri_filename)
    if not match:
        raise ValueError(f"Failed to parse MRI filename: {mri_filename}")
    
    prefix, patient_id, fraction_id,recon_id = match.groups()
    
    organ_masks = {
        'bladder': f"{prefix}_Bladder_{patient_id}_{fraction_id}_{recon_id}.mhd",
        'rectum': f"{prefix}_Rectum_{patient_id}_{fraction_id}_{recon_id}.mhd",
        'prostate': f"{prefix}_Prostate_{patient_id}_{fraction_id}_{recon_id}.mhd"
    }
    
    try:
        mri_image = ants.image_read(mri_path)
    except Exception as e:
        print(f"Failed to load MRI image: {str(e)}")
        mri_image = None
    
    masks = {}
    for organ, mask_filename in organ_masks.items():
        mask_path = os.path.join(mri_dir, mask_filename)
        try:
            if os.path.exists(mask_path):
                masks[organ] = ants.image_read(mask_path)
                #print(f"Successfully loaded {organ} mask: {mask_path}")
            else:
                masks[organ] = None
                print(f"{organ} mask not found: {mask_path}")
        except Exception as e:
            masks[organ] = None
            print(f"Failed to load {organ} mask: {str(e)}")
    
    return mri_image, masks['bladder'], masks['rectum'], masks['prostate']

def create_identity_dvf(reference_image):
    shape = reference_image.shape
    dimension = reference_image.dimension
    
    meshgrid = np.meshgrid(
        *[np.arange(s) for s in shape], 
        indexing='ij'
    )
    
    identity_dvf = []
    for i in range(dimension):
        displacement = np.zeros_like(meshgrid[i], dtype=np.float32)
        displacement_img = ants.from_numpy(displacement, origin=reference_image.origin,
                                         spacing=reference_image.spacing, 
                                         direction=reference_image.direction)
        identity_dvf.append(displacement_img)
    
    identity_dvf = ants.merge_channels(identity_dvf)
    return identity_dvf
def generate_dvf_by_points(transform, reference_image):
    shape = reference_image.shape
    origin = reference_image.origin
    spacing = reference_image.spacing
    dimension = reference_image.dimension
    
    displacement_array = np.zeros(shape + (dimension,), dtype=np.float32)
    
    total_points = np.prod(shape)
    with tqdm(total=total_points, desc="Generating DVF", unit="points") as pbar:
        for index in np.ndindex(shape):
            physical_point = [
                origin[i] + index[i] * spacing[i]
                for i in range(dimension)
            ]
            
            transformed_point = ants.apply_ants_transform_to_point(
                transform=transform,
                point=physical_point
            )
            
            displacement = [
                transformed_point[i] - physical_point[i]
                for i in range(dimension)
            ]
            
            displacement_array[index] = displacement
            pbar.update(1)
    
    dvf = ants.from_numpy(
        displacement_array,
        origin=origin,
        spacing=spacing,
        direction=reference_image.direction
    )
    
    return dvf    
def generate_dvf_by_points_MT(transform, reference_image, max_workers=None, chunk_size=100000):
    shape = reference_image.shape
    origin = reference_image.origin
    spacing = reference_image.spacing
    dimension = reference_image.dimension
    
    displacement_array = np.zeros(shape + (dimension,), dtype=np.float32)
    
    all_indices = [(i, j, k) for i in range(shape[0]) 
                  for j in range(shape[1]) 
                  for k in range(shape[2])] if dimension == 3 else \
                  [(i, j) for i in range(shape[0]) for j in range(shape[1])]
    
    def process_point(idx):
        physical_point = [origin[i] + idx[i] * spacing[i] for i in range(dimension)]
        transformed_point = ants.apply_ants_transform_to_point(transform, physical_point)
        displacement = [transformed_point[i] - physical_point[i] for i in range(dimension)]
        return idx, displacement
    
    def process_chunk(chunk_indices):
        results = []
        for idx in chunk_indices:
            results.append(process_point(idx))
        return results
    
    num_chunks = (len(all_indices) + chunk_size - 1) // chunk_size
    chunks = [all_indices[i*chunk_size : (i+1)*chunk_size] for i in range(num_chunks)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_list = list(tqdm(
            executor.map(process_chunk, chunks),
            total=num_chunks,
            desc="Generating DVF"
        ))
    
    for results in results_list:
        for idx, displacement in results:
            displacement_array[idx] = displacement
    
    dvf = ants.from_numpy(
        displacement_array,
        origin=origin,
        spacing=spacing,
        direction=reference_image.direction
    )
    
    return dvf  
def _process_chunk(args):
    chunk, origin, spacing, transform_file1, transform_file2 = args
    transform1 = ants.read_transform(transform_file1)
    transform2 = ants.read_transform(transform_file2)
    vec = ants.image_read( transform_file2 )
    transform2 = ants.transform_from_displacement_field( vec )
    composite_transform = ants.compose_transforms([transform1, transform2])
    physical_points = origin + chunk * spacing
    transformed_points = np.array([
        ants.apply_ants_transform_to_point(composite_transform, pt)
        for pt in physical_points
    ])
    displacements = transformed_points - physical_points
    return chunk.astype(int), displacements

def generate_dvf_by_points_MP(transform_file1, transform_file2, reference_image, 
                             max_workers=None, chunk_size=10000):
    shape = reference_image.shape
    origin = np.array(reference_image.origin)
    spacing = np.array(reference_image.spacing)
    dimension = reference_image.dimension
    displacement_array = np.zeros(shape + (dimension,), dtype=np.float32)
    grid = np.stack(np.mgrid[[slice(0, s) for s in shape]], axis=-1)
    chunks = np.array_split(grid.reshape(-1, dimension), 
                          grid.size // dimension // chunk_size + 1)
    params = [(chunk, origin, spacing, transform_file1, transform_file2) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_chunk, param) for param in params]
        for future in tqdm(futures, desc="Processing chunks"):
            indices, displacements = future.result()
            displacement_array[tuple(indices.T)] = displacements
    return ants.from_numpy(
        displacement_array,
        origin=origin,
        spacing=spacing,
        direction=reference_image.direction
    )    

def generate_dvf_by_batch(transform, reference_image, batch_size=10000):
    shape = reference_image.shape
    origin = reference_image.origin
    spacing = reference_image.spacing
    dimension = reference_image.dimension
    
    total_points = np.prod(shape)
    all_indices = np.array([index for index in np.ndindex(shape)])
    
    displacement_array = np.zeros(shape + (dimension,), dtype=np.float32)
    
    for i in range(0, total_points, batch_size):
        batch_indices = all_indices[i:i+batch_size]
        
        physical_points = np.array([
            [origin[j] + idx[j] * spacing[j] for j in range(dimension)]
            for idx in batch_indices
        ])
        
        points_image = ants.from_numpy(
            physical_points.reshape((len(batch_indices),) + (1,) * (dimension-1) + (dimension,)),
            origin=reference_image.origin,
            spacing=reference_image.spacing,
            direction=reference_image.direction
        )
        
        transformed_points_image = ants.apply_transforms(
            fixed=reference_image,
            moving=points_image,
            transformlist=[transform]
        )
        
        transformed_points = transformed_points_image.numpy().reshape(-1, dimension)
        
        for j, idx in enumerate(batch_indices):
            displacement = transformed_points[j] - physical_points[j]
            displacement_array[tuple(idx)] = displacement
    
    dvf = ants.from_numpy(
        displacement_array,
        origin=origin,
        spacing=spacing,
        direction=reference_image.direction
    )
    
    return dvf 

def batch_transform_points(transform, points):
    dimension = points.shape[1]
    points_image = ants.from_numpy(
        points.reshape((len(points),) + (1,) * (dimension-1) + (dimension,)),
        origin=[0.0] * dimension,
        spacing=[1.0] * dimension,
        direction=np.eye(dimension)
    )
    
    transformed_points_image = ants.apply_transforms(
        fixed=points_image,
        moving=points_image,
        transformlist=[transform],
        interpolator='linear'
    )
    
    return transformed_points_image.numpy().reshape(-1, dimension)   

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
def main():

    

    """Main function"""
    # Set data paths
    data_folder = "D:\\ProstatePatientData"  # Root directory of patient data
    output_folder = "D:\\ProstatePatientOutData" # Root directory for output results
    
    # Step 1: Collect all setup MRIs and calculate bladder volumes
    print("Collecting all setup MRIs and calculating bladder volumes...")
    all_setup_mris = []
    
    # Get all patient folders
    patient_folders = [f for f in os.listdir(data_folder) 
                      if os.path.isdir(os.path.join(data_folder, f))]
    
    # Iterate through all patients and fractions to collect setup MRIs
    for patient_folder in tqdm(patient_folders, desc="Processing patients"):
        patient_id = os.path.basename(patient_folder)
        patient_path = os.path.join(data_folder, patient_folder)
        
        # Get all fraction folders
        fraction_folders = [f for f in os.listdir(patient_path) 
                           if os.path.isdir(os.path.join(patient_path, f))]
        
        for fraction_folder in fraction_folders:
            fraction_id = os.path.basename(fraction_folder)
            fraction_path = os.path.join(patient_path, fraction_folder)
            
            # Find setup MRI image
            setup_mri_path = find_setup_mri(fraction_path, patient_id, fraction_id)
            if not setup_mri_path:
                print(f"    Warning: Setup MRI not found in {patient_id}/{fraction_id}")
                continue
            
            # Find bladder mask
            bladder_mask_path = find_mask(fraction_path, patient_id, fraction_id, "bladder")
            if not bladder_mask_path:
                print(f"    Warning: Bladder mask not found in {patient_id}/{fraction_id}")
                continue
            
            # Load bladder mask and calculate volume
            bladder_mask = load_mask(bladder_mask_path)
            if bladder_mask is None:
                continue
            
            volume = calculate_volume(bladder_mask)
            all_setup_mris.append({
                'patient_id': patient_id,
                'fraction_id': fraction_id,
                'setup_mri_path': setup_mri_path,
                'bladder_volume': volume
            })
    
    # Check if any valid setup MRIs were found
    if not all_setup_mris:
        print("No valid setup MRIs found!")
        return
    
    # Step 2: Select reference setup MRI based on median bladder volume
    all_setup_mris.sort(key=lambda x: x['bladder_volume'])
    median_index = len(all_setup_mris) // 2
    reference_setup_info = all_setup_mris[median_index]
    
    print(f"Selected reference setup MRI:")
    print(f"  Patient ID: {reference_setup_info['patient_id']}")
    print(f"  Fraction ID: {reference_setup_info['fraction_id']}")
    print(f"  Bladder Volume: {reference_setup_info['bladder_volume']:.2f} mm^3")
    
    # Load reference setup MRI
    reference_setup_mri,reference_mask_bladder, reference_mask_rectum, reference_mask_prostate = load_mri_with_masks(reference_setup_info['setup_mri_path'])
    reference_bladdermask_arr=reference_mask_bladder.numpy()
    reference_prostatemask_arr=reference_mask_prostate.numpy()
    reference_rectummask_arr=reference_mask_rectum.numpy()
    reference_arr = np.logical_or(reference_bladdermask_arr, reference_prostatemask_arr,reference_rectummask_arr).astype(np.uint8)

    
    if reference_setup_mri is None:
        print(f"Failed to load reference setup MRI: {reference_setup_info['setup_mri_path']}")
        return
    
    # Step 3: Process each patient and register their setup MRIs to the reference
    print("Processing each patient and registering setup MRIs to reference...")
    for patient_folder in tqdm(patient_folders, desc="Processing patients"):
        patient_id = os.path.basename(patient_folder)
        patient_path = os.path.join(data_folder, patient_folder)
        
        # Get all fraction folders
        fraction_folders = [f for f in os.listdir(patient_path) 
                           if os.path.isdir(os.path.join(patient_path, f))]
        
        for fraction_folder in fraction_folders:
            fraction_id = os.path.basename(fraction_folder)
            fraction_path = os.path.join(patient_path, fraction_folder)
            
            # Skip the reference fraction
            if (patient_id == reference_setup_info['patient_id'] and 
                fraction_id == reference_setup_info['fraction_id']):
                print(f"Skipping reference fraction: {patient_id}/{fraction_id}")
                continue
            
            # Find setup MRI image
            setup_mri_path = find_setup_mri(fraction_path, patient_id, fraction_id)
            if not setup_mri_path:
                print(f"    Warning: Setup MRI not found in {patient_id}/{fraction_id}")
                continue
            
            # Load setup MRI
            setup_mri,setup_mask_bladder, setup_mask_rectum, setup_mask_prostate = load_mri_with_masks(setup_mri_path)
            setup_bladdermask_arr=setup_mask_bladder.numpy()
            setup_prostatemask_arr=setup_mask_prostate.numpy()
            setup_rectummask_arr=setup_mask_rectum.numpy()
            setup_arr = np.logical_or(setup_bladdermask_arr, setup_prostatemask_arr,setup_rectummask_arr).astype(np.uint8)
            mask_arr =np.logical_or(reference_arr, setup_arr).astype(np.uint8)
            mask = ants.from_numpy(mask_arr)
            mask.set_spacing(setup_mri.spacing)
            mask.set_origin(setup_mri.origin)
            mask.set_direction(setup_mri.direction)
            if setup_mri is None:
                continue
            
            print(f"Registering setup MRI from {patient_id}/{fraction_id} to reference")
            
            # Create output directory
            output_registration_folder = os.path.join(
                output_folder, patient_id, f"registration_to_refernce_{patient_id}_{fraction_id}")
            os.makedirs(output_registration_folder, exist_ok=True)
            
            # Step 1: Rigid registration
            #rigid_output_prefix = os.path.join(
            #    output_registration_folder, 
            #    f"rigid_f{fraction_id}_to_ref")
            
            rigid_registration = perform_rigid_registration(
                fixed_image=reference_setup_mri,
                moving_image=setup_mri
            )
            
            # Save rigid transform
            #rigid_transforms = rigid_registration['fwdtransforms']
            
            
            # Step 2: Deformable registration with rigid initial transform
            deformable_output_prefix = os.path.join(
                output_registration_folder, 
                f"DVF_{patient_id}_{fraction_id}_to_ref")
            
            deformable_registration = perform_deformable_registration(
                fixed_image=reference_setup_mri,
                moving_image=setup_mri,
                mask=mask,
                initial_transform=rigid_registration
            )
            
            # Save deformable transform
            #deformable_transforms = deformable_registration['fwdtransforms']
            #rigid_transform=deformable_registration["fwdtransforms"][0]
            #vec = ants.image_read( deformable_registration['fwdtransforms'][0] )
            #deformable_transform = ants.transform_from_displacement_field( vec )
            #rigid_transform=ants.read_transform(rigid_registration["fwdtransforms"][0])
            filelist=[]
            filelist.append(rigid_registration["fwdtransforms"][0])
            filelist.append(deformable_registration["fwdtransforms"][0])
            composed_transform=generate_dvf_by_points_new(filelist, setup_mri)

            compose_output_path = os.path.join(
                output_registration_folder, 
                f"composedDVF_{patient_id}_{fraction_id}_to_ref.mhd")
            
           
            ants.image_write(composed_transform, compose_output_path)
            print(f"Composed transform saved to: {compose_output_path}")
            
            # Process reconstructed MRIs for this fraction
            process_reconstructed_mris(
                patient_folder=patient_path,
                patient_id=patient_id,
                fraction_id=fraction_id,
                reference_setup_mri=reference_setup_mri,
                reference_fraction_id=reference_setup_info['fraction_id'],
                setup_to_reference_transform_path=compose_output_path,
                output_base_folder=output_folder
            )


def process_reconstructed_mris(patient_folder, patient_id, fraction_id, 
                               reference_setup_mri, reference_fraction_id,
                               setup_to_reference_transform_path, output_base_folder):
    """
    Process reconstructed MRIs for a given fraction and create new composite transforms
    
    Args:
    patient_folder: Path to the patient folder
    patient_id: Patient ID
    fraction_id: Fraction ID
    reference_setup_mri: Reference setup MRI image
    reference_fraction_id: Reference fraction ID
    setup_to_reference_transform: Transform from current setup to reference setup
    output_base_folder: Path to the output base folder
    """
    fraction_path = os.path.join(patient_folder, fraction_id)
    
    # Find all reconstructed MRI images for this fraction
    reconstructed_mris = find_reconstructed_mris(fraction_path, patient_id, fraction_id)
    if not reconstructed_mris:
        print(f"    No reconstructed MRIs found in fraction {fraction_id}")
        return
    
    # Find setup MRI for this fraction
    setup_mri_path = find_setup_mri(fraction_path, patient_id, fraction_id)
    if not setup_mri_path:
        print(f"    Setup MRI not found in fraction {fraction_id}")
        return
    
    #setup_mri = load_mri_image(setup_mri_path)
    setup_mri,setup_mask_bladder, setup_mask_rectum, setup_mask_prostate = load_mri_with_masks(setup_mri_path)
    if setup_mri is None:
        return
    setup_bladdermask_arr=setup_mask_bladder.numpy()
    setup_prostatemask_arr=setup_mask_prostate.numpy()
    setup_rectummask_arr=setup_mask_rectum.numpy()
    setup_arr = np.logical_or(setup_bladdermask_arr, setup_prostatemask_arr,setup_rectummask_arr).astype(np.uint8)
            
    
    # Create output directory for registration results

    output_registration_folder = os.path.join(
                output_base_folder, patient_id, f"recon_to_setup_registration{patient_id}_{fraction_id}")
    os.makedirs(output_registration_folder, exist_ok=True)
    
   
    
    # Invert the setup_to_reference transform
    #inverted_setup_to_reference_transform = invert_transform(setup_to_reference_transform)
    
    # Process each reconstructed MRI
    for recon_mri_path in reconstructed_mris:
        recon_mri_name = os.path.basename(recon_mri_path)
        recon_id = re.search(r'Recon_{0}_{1}_(\d+)\.mhd'.format(patient_id, fraction_id), recon_mri_name).group(1)
        print(f"  Processing reconstructed MRI: {recon_mri_name}")
        
        # Load reconstructed MRI
        recon_mri,recon_mask_bladder, recon_mask_rectum, recon_mask_prostate = load_reconmri_with_masks(recon_mri_path)
        if recon_mri is None:
            continue

        recon_bladdermask_arr=recon_mask_bladder.numpy()
        recon_prostatemask_arr=recon_mask_prostate.numpy()
        recon_rectummask_arr=recon_mask_rectum.numpy()
        recon_arr = np.logical_or(recon_bladdermask_arr, recon_prostatemask_arr,recon_rectummask_arr).astype(np.uint8)

        mask_arr =np.logical_or(recon_arr, setup_arr).astype(np.uint8)
        mask = ants.from_numpy(mask_arr)
        mask.set_spacing(setup_mri.spacing)
        mask.set_origin(setup_mri.origin)
        mask.set_direction(setup_mri.direction)
        
        # Perform deformable registration from reconstructed MRI to setup MRI
        print(f"    Registering reconstructed MRI to setup MRI for fraction {fraction_id}")

        deformable_registration = perform_deformable_registration(
                fixed_image=recon_mri,
                moving_image=setup_mri,
                mask=mask,
                initial_transform=None
            )
        filelist=[]
        
        filelist.append(deformable_registration["fwdtransforms"][0])
        filelist.append(setup_to_reference_transform_path)
        composed_transform=generate_dvf_by_points_new(filelist, setup_mri)
        
       
        composite_output_path = os.path.join(
            output_registration_folder, 
            f"compositeDVF_recon{recon_id}_f{fraction_id}_to_ref.mhd")
        
        ants.image_write(composed_transform, composite_output_path)
        print(f"    Composite transform saved to: {composite_output_path}")



if __name__ == "__main__":
    main()    