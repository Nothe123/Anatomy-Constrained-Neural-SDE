
![image](https://github.com/user-attachments/assets/03aff302-666b-4929-b2ed-0fec34cb3929)


The codes are  implementaion of "A Probabilistic organ motion model based on Anatomy-Constrained Neural Stochastic Differential Equation for Prostate MRI-guided Radiotherapy"
The general work flow was shown above. The code consists of four main modules:
1. 3DMRIReconstruction.py was used to establish and apply the deep learning model for 3D-MRI reocnstruciton from single cine-MRI.
2. RegistrationScript.py performed the rigid and deformable image registraiotn to transform the organ motion from different patients to the same anatomy space
3. IncrementalPCA redcuded the dimension of DVF generated from registraiton to simplify the problem
4. NeuralSDEwithConstraints constructed and applied the anatomy constrained neural-SDE
The neural-SDE was constructed by executing the above  modules in order.

Any question about the code, you can contact via wei_cn00@163.com
                        
