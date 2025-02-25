import torch


class SVD_Helper():
    def __init__(self, m, n):
        self.U = torch.rand(m,m, requires_grad=True)
        self.S = torch.rand(m,n, requires_grad=True)
        self.V_T = torch.rand(n,n, requires_grad=True)

       
    def SVD(self):
        return self.U, self.S, self.V_T
    

    def train(self, X, step_size=5e-5, no_steps=10000, tol=1e-4):
        log_freq = 100

        optimizer = torch.optim.SGD([self.U, self.S , self.V_T], lr=step_size)

        for i in range(no_steps):
                        
            m1 = torch.matmul(self.S, self.V_T.T)
            m2 = torch.matmul(self.U, m1)
            term = X - m2
            loss = torch.linalg.matrix_norm(term, ord='fro')
            loss = torch.sqrt(loss)
            
            # Backprop & Update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % log_freq == 0:
                print('Loss is %s at iteration %i' % (loss, i))

            if abs(loss) <= tol:
                break
