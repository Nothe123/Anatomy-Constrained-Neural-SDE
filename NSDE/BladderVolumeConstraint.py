
import torch
class BladderVolumeConstraint:
    def __init__(self, ref_blader, mean, vectors,input_size,threshold=0.1):
        
        self.ref_bladder=ref_blader

        
        self.D = ref_blader.shape[0]
        self.H = ref_blader.shape[1]
        self.W = ref_blader.shape[2]
        self.mean = mean
        self.vectors = vectors
        self.z = torch.linspace(-1, 1, self.D)
        self.y = torch.linspace(-1, 1, self.H)
        self.x = torch.linspace(-1, 1, self.W)
        self.Threshold = threshold
        self.grid_z, self.grid_y, self.grid_x = torch.meshgrid(self.z, self.y, self.x, indexing='ij')
        self.base_grid = torch.stack([self.grid_x, self.grid_y, self.grid_z], dim=-1).unsqueeze(0).to(mean.device)
        self.pca_coeffs=torch.zeros(input_size,device=vectors.device)
        self.pca_coeffs.requires_grad = True
    def SetPCACoefficients(self,PCAcoefficient):
        self.pca_coeffs=PCAcoefficient.clone().detach().requires_grad_(True)
        
    def generate_3d_dvf(self):
        batch_size = self.pca_coeffs.shape[0]
        num_components = self.pca_coeffs.shape[1] 
        a_expanded = self.pca_coeffs.view(batch_size, num_components, 1, 1, 1, 1)
        b_expanded = self.vectors.unsqueeze(0)  
        component_dvf = a_expanded * b_expanded
        dvf = component_dvf.sum(dim=1)
        del a_expanded, b_expanded, component_dvf
        torch.cuda.empty_cache() if dvf.is_cuda else None

        mean_expanded = self.mean.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        dvf += mean_expanded
        return dvf
    
    def calculate_bladdervolumeVSPCAgradients_old(self):
        batch_size = self.pca_coeffs.shape[0]
        dvf_3d = self.generate_3d_dvf()
        deformed_grid = self.base_grid + dvf_3d.permute(0, 2, 3, 4, 1)

        ref_bladder_expand=self.ref_bladder.unsqueeze(0).expand(batch_size,-1,-1,-1,-1)
        deformed_bladder= torch.nn.functional.grid_sample(ref_bladder_expand, deformed_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        threshold = 0.1
        V = torch.sum(torch.sigmoid(deformed_bladder / threshold), dim=(1, 2, 3, 4))
        grads_batchlist = torch.zeros_like(self.pca_coeffs)
        for i in range(batch_size):
            single_V = V[i]  
            grads = torch.autograd.grad(single_V, self.pca_coeffs, retain_graph=True if i < batch_size - 1 else False)
            gradstensor=grads[0]
            grads_batchlist[i,:]= gradstensor[i,:]
        
        return grads_batchlist
    def calculate_bladdervolumeVSPCAgradients(self):
        all_grads = []
        batch_size = self.pca_coeffs.shape[0]
        dvf_3d = self.generate_3d_dvf()
        deformed_grid = self.base_grid + dvf_3d
        ref_bladder_expand = self.ref_bladder.expand(batch_size, *self.ref_bladder.shape).unsqueeze(1)
        deformed_bladder = torch.nn.functional.grid_sample(ref_bladder_expand, deformed_grid,mode='bilinear',padding_mode='zeros',align_corners=True)
        threshold = 0.1
        V = torch.sum(torch.sigmoid(deformed_bladder / threshold), dim=(1, 2, 3, 4))
        grad_outputs = torch.ones_like(V) 
        grads = torch.autograd.grad(outputs=V,inputs=self.pca_coeffs,grad_outputs=grad_outputs,retain_graph=False, create_graph=False)[0]
        del dvf_3d, deformed_grid, ref_bladder_expand, deformed_bladder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return grads


    def find_orthogonal_complement_basis(self,A):
        batch_size = self.pca_coeffs.shape[0]

        k = A.shape[0]
        basis_matrix = torch.zeros(k, k)
        basis_matrix[0] = A
        standard_basis = torch.eye(k)

        for i in range(1, k):
            v = standard_basis[i]
            for j in range(i):
                proj = torch.dot(v, basis_matrix[j]) / torch.dot(basis_matrix[j], basis_matrix[j]) * basis_matrix[j]
                v -= proj
        if torch.norm(v) > 1e-10:
            basis_matrix[i] = v / torch.norm(v)
    
            orthogonal_basis = basis_matrix[1:]
        return orthogonal_basis

    def generate_solution_space_basis(self,A, C, num_basis=None):

        if num_basis is None:
            num_basis = A.shape[0] - 1
        x_0 = (C / torch.dot(A, A)) * A
        orthogonal_basis = self.find_orthogonal_complement_basis(A)

        basis_vectors = []
        for i in range(num_basis):
            coef = torch.zeros(orthogonal_basis.shape[0])
            coef[i % orthogonal_basis.shape[0]] = 1.0
            y = torch.sum(coef.unsqueeze(1) * orthogonal_basis, dim=0)
            scale_factor = torch.relu(torch.dot(A, y)) / torch.dot(A, y)
            y = scale_factor * y
            x = x_0 + y
            basis_vectors.append(x)
        basis_vectors = torch.stack(basis_vectors)
        return basis_vectors

   
    def run(self):
        bladderVgradient = self.calculate_bladdervolumeVSPCAgradients()
        #solutionbasis=self.generate_solution_space_basis(bladderVgradient, self.Threshold)
        return bladderVgradient
