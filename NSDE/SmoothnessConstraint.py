
import torch
class SmoothnessConstraint:
    def __init__(self, ref_image,  mean, vectors,input_size):
        
        self.ref_blader=ref_image

        
        self.D = ref_image.shape[0]
        self.H = ref_image.shape[1]
        self.W = ref_image.shape[2]
        self.vectors = vectors
        self.mean = mean
        self.z = torch.linspace(-1, 1, self.D)
        self.y = torch.linspace(-1, 1, self.H)
        self.x = torch.linspace(-1, 1, self.W)
        self.grid_z, self.grid_y, self.grid_x = torch.meshgrid(self.z, self.y, self.x, indexing='ij')
        self.base_grid = torch.stack([self.grid_x, self.grid_y, self.grid_z], dim=-1).unsqueeze(0).to(vectors.device)
        self.pca_coeffs=torch.zeros(input_size,device=vectors.device)
        self.pca_coeffs.requires_grad = True
    def SetPCAcoefficients(self,PCAcoefficients):
        self.pca_coeffs=PCAcoefficients.clone().detach().requires_grad_(True)
        
    def generate_3d_dvf(self):
        batch_size, seq_len, num_components = self.pca_coeffs.shape
        
        # Reshape to (batch*seq_len, num_components)
        coeff_flat = self.pca_coeffs.reshape(-1, num_components)
        
        # Expand dimensions for broadcasting
        a_expanded = coeff_flat.view(-1, num_components, 1, 1, 1, 1)  # (batch*seq_len, num_components, 1,1,1,1)
        b_expanded = self.vectors.unsqueeze(0)                          # (1, num_components, D,H,W,3)
        
        # Compute component DVF
        component_dvf = a_expanded * b_expanded  # (batch*seq_len, num_components, D,H,W,3)
        dvf = component_dvf.sum(dim=1)            # (batch*seq_len, D,H,W,3)
        
        # Add mean and reshape back
        mean_expanded = self.mean.unsqueeze(0).expand(dvf.shape[0], -1, -1, -1, -1)
        dvf += mean_expanded
        dvf = dvf.view(batch_size, seq_len, self.D, self.H, self.W, 3)  # (batch, seq_len, D,H,W,3)
        #print(dvf.shape)
        
        return dvf

        
    def calculate_DVFvstimegradients(self):
        dvf = self.generate_3d_dvf()
        d_dvf_dt = torch.zeros_like(dvf)
    
        
        d_dvf_dt[:, 1:-1] = (dvf[:, 2:] - dvf[:, :-2]) / 2.0 
    
        
        d_dvf_dt[:, 0] = dvf[:, 1] - dvf[:, 0]    
        d_dvf_dt[:, -1] = dvf[:, -1] - dvf[:, -2] 
    
        return d_dvf_dt   
       


   
    def run(self,PCAgriadients):
        DVFvsPCAgradients = self.calculate_DVFvstimegradients()
        grad_norm = torch.norm(DVFvsPCAgradients, p=2, dim=-1)  # (batch, seq_len)
        TotalGradient = grad_norm.mean() 
        return TotalGradient


