import torch

class SVD_Helper():
    def __init__(self, n, m):        
        # U should be m x r (not m x m)
        self.U = torch.nn.init.orthogonal_(torch.empty(m, m)).float().requires_grad_()
        
        # V should be n x r (not n x n)
        self.V = torch.nn.init.orthogonal_(torch.empty(n,n)).float().requires_grad_()
        
        # S should be r x r (not m x n)
        self.S = torch.nn.Parameter(torch.eye(m,n) * torch.rand(m,n))  # Singular values
        #

    def SVD(self):
        return self.U, self.S, self.V

    def train(self, X, step_size=5e-4, no_steps=5000, tol=1e-6):
        log_freq = 250
        optimizer = torch.optim.Adam([self.U, self.S, self.V], lr=step_size)

        for i in range(no_steps):
            # Reconstruct X
            X_approx = self.U @ self.S @ self.V.T  

            # Orthonormality constraints
            I_U = torch.eye(self.U.shape[1])  # Identity for U^T U
            I_V = torch.eye(self.V.shape[1])  # Identity for V^T V
            
            ortho_U = torch.norm(self.U.T @ self.U - I_U, p='fro')
            ortho_V = torch.norm(self.V.T @ self.V - I_V, p='fro')            

            reconstruction_error = torch.norm(X - X_approx, p="fro")

            loss = reconstruction_error #+ ortho_U + ortho_V

            # Backprop & Update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % log_freq == 0:
                print(f'Loss: {loss.item()} at iteration {i}')

            if loss.item() <= tol:
                break
            

