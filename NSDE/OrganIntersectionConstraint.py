import torch
import SimpleITK as sitk
initial_allocated = torch.cuda.memory_allocated()
initial_reserved = torch.cuda.memory_reserved()
class OrganIntersectionConstraint:
    def __init__(self, ref_blader,ref_rectum,ref_prostate, mean, vectors,input_size):
        self.num_organs = 3
        self.ref_blader=ref_blader
        self.ref_rectum=ref_rectum
        self.ref_prostate=ref_prostate
        
        self.D = ref_blader.shape[0]
        self.H = ref_blader.shape[1]
        self.W = ref_blader.shape[2]
        self.vectors = vectors
        self.mean = mean
        self.z = torch.linspace(-1, 1, self.D)
        self.y = torch.linspace(-1, 1, self.H)
        self.x = torch.linspace(-1, 1, self.W)
        self.grid_z, self.grid_y, self.grid_x = torch.meshgrid(self.z, self.y, self.x, indexing='ij')
        self.base_grid = torch.stack([self.grid_x, self.grid_y, self.grid_z], dim=-1).unsqueeze(0).to(vectors.device)
        self.organs = []
        self.organs.append(self.ref_blader)
        self.organs.append(self.ref_rectum)
        self.organs.append(self.ref_prostate)
        self.pca_coeffs=torch.zeros(input_size,device=vectors.device)
        self.pca_coeffs.requires_grad = True
    def SetPCACoefficients(self,PCAcoefficient):
        self.pca_coeffs=PCAcoefficient.clone().detach().requires_grad_(True)
        #print(self.pca_coeffs.shape)
        
      


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
    def generate_3d_dvf_old(self):
        batch_size = self.pca_coeffs.shape[0]
        dvf = torch.zeros((batch_size, 3, self.D, self.H, self.W), device=self.pca_coeffs.device)
        
        for i in range(len(self.pca_coeffs[0])):
            a=self.pca_coeffs[:, i].unsqueeze(1)
            b=self.vectors[i].unsqueeze(0)
            a_expanded = a.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            c = a_expanded * b
            c=c.squeeze(1)
            d= torch.permute(c,(0,4,1,2,3))
            dvf += d
        mean_expanded = self.mean.unsqueeze(0).expand(batch_size,-1,-1,-1,-1)
        mean_expanded_p=torch.permute(mean_expanded,(0,4,1,2,3))
        dvf += mean_expanded_p
        return dvf

    def calculate_gradients(self):
        all_grads = []
        batch_size = self.pca_coeffs.shape[0]
        organ_pairs = [(i, j) for i in range(self.num_organs) for j in range(i+1, self.num_organs)]
        
        for i, j in organ_pairs:
            n_A = self.organs[i]
            n_B = self.organs[j]
        
            #with torch.no_grad():  
            dvf_3d = self.generate_3d_dvf()
            
        
            
            #print(ss.shape)
            #print(dvf_3d.shape)
            deformed_grid = self.base_grid + dvf_3d#.permute(0, 2, 3, 4, 1)
            deformed_grid.requires_grad_(True)
            #print(deformed_grid.shape)
        
        
            n_A_expand = n_A.expand(batch_size, *n_A.shape).unsqueeze(1)
            n_B_expand = n_B.expand(batch_size, *n_B.shape).unsqueeze(1)
            

            mA = torch.nn.functional.grid_sample(n_A_expand, deformed_grid,mode='bilinear',padding_mode='zeros',align_corners=True)
            mB = torch.nn.functional.grid_sample(n_B_expand, deformed_grid,
                                            mode='bilinear',
                                            padding_mode='zeros',
                                            align_corners=True)
           
        

            threshold = 0.1
            product = mA * mB
            
            V = torch.sum(torch.sigmoid(product / threshold), dim=(1, 2, 3, 4))
            
        
            #grad_outputs = torch.eye(batch_size, device=V.device)
            grad_outputs = torch.ones_like(V) 

           
            

            grads = torch.autograd.grad(outputs=V,inputs=self.pca_coeffs,grad_outputs=grad_outputs,retain_graph=False, create_graph=False)[0]
        
            all_grads.append(grads)
        
        
            del dvf_3d, deformed_grid, n_A_expand, n_B_expand, mA, mB, product
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
        return all_grads
        



    
    def calculate_gradients_old(self):
        all_grads = []
        batch_size = self.pca_coeffs.shape[0]
        for i in range(self.num_organs):
            for j in range(i + 1, self.num_organs):
                n_A = self.organs[i]
                n_B = self.organs[j]
                dvf_3d = self.generate_3d_dvf_old()
                print(dvf_3d.shape)
                deformed_grid = self.base_grid + dvf_3d.permute(0, 2, 3, 4, 1)
                print(self.base_grid.shape)
                print(deformed_grid.shape)
                n_A_expand=n_A.unsqueeze(0).expand(batch_size,-1,-1,-1,-1)
                n_B_expand=n_B.unsqueeze(0).expand(batch_size,-1,-1,-1,-1)
                mA = torch.nn.functional.grid_sample(n_A_expand, deformed_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                mB = torch.nn.functional.grid_sample(n_B_expand, deformed_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

                threshold = 0.1
                product = mA*mB
                V = torch.sum(torch.sigmoid(product / threshold), dim=(1, 2, 3, 4))
                grads_batchlist = torch.zeros_like(self.pca_coeffs)
                for i in range(batch_size):
                    single_V = V[i]  
                    
                   
                    grads = torch.autograd.grad(single_V, self.pca_coeffs, retain_graph=True if i < batch_size - 1 else False)
                    gradstensor=grads[0]
                    grads_batchlist[i,:]= gradstensor[i,:]
                #print(grads_batchlist)
                all_grads.append(grads_batchlist)
        return all_grads
    def calculate_nullspace_basis_old(self, all_grads):
        batch_size = self.pca_coeffs.shape[0]
        PCA_length = self.pca_coeffs.shape[1]
        nullspace_bases=[]
        if batch_size > 1:
            
            batch_grads_in = torch.zeros(3,PCA_length,device=self.pca_coeffs.device)
            for i in range(batch_size):
                
                batch_grads_in[0,:] = all_grads[0][i,:]
                batch_grads_in[1,:] = all_grads[1][i,:]
                batch_grads_in[2,:] = all_grads[2][i,:]
                batch_nullspace_basis = self._calculate_single_batch_nullspace_basis_old(batch_grads_in)
                nullspace_bases.append(batch_nullspace_basis)
            
            #nullspace_bases.append(nullspace_bases)# = torch.stack(nullspace_bases)
            return nullspace_bases
        else:
            return self._calculate_single_batch_nullspace_basis(all_grads)
    def _calculate_single_batch_nullspace_basis_old(self, all_grads):
        U, S, Vh = torch.linalg.svd(all_grads)
        tol = 1e-5
        rank = torch.sum(S > tol)
        null_space_dim = all_grads.shape[1] - rank 
        #null_basis=torch.zeros([all_grads.shape[0],)
        Vh[1,:]=0
        #null_basis = Vh[rank:, :].T  
        null_basis2 = Vh.T  
        return null_basis2
    def calculate_nullspace_basis(self, all_grads):
        batch_size = self.pca_coeffs.shape[0]
        PCA_length = self.pca_coeffs.shape[1]
        stacked_grads = torch.stack([g for g in all_grads], dim=0)  # [3, batch_size, PCA_length]
        stacked_grads = stacked_grads.permute(1, 0, 2)  # [batch_size, 3, PCA_length]
        
            
        nullspace_bases = torch.zeros([batch_size,PCA_length,PCA_length],device= self.pca_coeffs.device)
        index=0
        for batch_grads in stacked_grads:
            basis = self._calculate_single_batch_nullspace_basis_old(batch_grads)
            nullspace_bases[index,:,:]=basis
            
        
        del stacked_grads, basis
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return nullspace_bases
    
        

    def _calculate_single_batch_nullspace_basis(self, grads_matrix):
    
        U, S, Vh = torch.linalg.svd(grads_matrix, full_matrices=False)
        tol = 1e-5
    
   
        rank = torch.sum(S > (S[0] * tol))  
        null_space_dim = grads_matrix.shape[1] - rank
    
        
        if null_space_dim > 0:
            null_basis = Vh[rank:].T  # [PCA_length, null_space_dim]
        else:
            null_basis = torch.zeros((grads_matrix.shape[1], 0), device=grads_matrix.device)
    
    
        Q, _ = torch.linalg.qr(null_basis, mode='reduced')
        return Q
    def run(self):
        all_grads = self.calculate_gradients()
        nullspace_basis = self.calculate_nullspace_basis(all_grads)
        return nullspace_basis
