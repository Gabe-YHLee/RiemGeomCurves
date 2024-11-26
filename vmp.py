import torch

class VMP(object):
    def __init__(
        self, 
        b: int=30, 
        h_mul: float=1, 
        dim: int=2, 
        via_points: list[torch.tensor, torch.tensor]=None, 
        num_curves: int=1, 
        device: str='cpu'
    ):
        self.b = b
        self.h_mul = h_mul
        self.dim = dim
        self.via_points = via_points
        self.weight = torch.autograd.Variable(torch.randn(num_curves, b, dim).to(device), requires_grad=True)
        self.set_Riemannian_metric()

    def __call__(
        self,
        z: torch.tensor,
    ) -> torch.tensor:
        '''
        z: (bs, L, 1)
        '''
        basis_values = self.Gaussian_basis(z)
        phi_values = self.phi(basis_values)
        return self.vbf(z, phi_values)
    
    def set_w(self, w: torch.tensor):
        self.weight = w

    def Gaussian_basis(self, z: torch.tensor) -> torch.tensor:
        '''r
        z: (bs, L, 1)
        '''
        c = torch.linspace(0, 1, self.b).to(z)
        h = self.h_mul*1/(self.b-1)
        values = torch.exp(-(z - c.view(1, 1, -1))**2/h**2)
        return values # (bs, L, b)
    
    def phi(self, basis_values: torch.tensor) -> torch.tensor:
        return basis_values/basis_values.sum(dim=-1, keepdim=True) # (bs, L, b)

    def vbf(self, z: torch.tensor, phi_values: torch.tensor) -> torch.tensor:
        '''
        z: (bs, L, 1)
        phi_values: (bs, L, b)
        '''
        init = self.via_points[0].clone().detach().to(z).view(1, 1, -1) # (1, 1, dim)
        final = self.via_points[1].clone().detach().to(z).view(1, 1, -1) # (1, 1, dim)
        return z*(final) + (1-z)*(init) + (z)*(1-z)*phi_values@self.weight

    def LfD(self, trajs: torch.tensor) -> torch.tensor:
        '''
        trajs: (bs, L, dim)
        '''
        bs, L, dim = trajs.size()
        z = torch.linspace(0, 1, L).view(1, -1, 1).to(trajs) # (1, L, 1)
        basis_values = self.Gaussian_basis(z) # (1, L, b)
        Phi =self.phi(basis_values).view(1, L, -1) # (1, L, b)

        init = torch.tensor(self.via_points[0], dtype=torch.float32).to(trajs).view(1, 1, dim) # (1, 1, dim)
        final = torch.tensor(self.via_points[-1], dtype=torch.float32).to(trajs).view(1, 1, dim) # (1, 1, dim)
        
        Phi = z*(1-z)*Phi
        return torch.linalg.pinv(Phi)@(trajs - z*(final) - (1-z)*(init))

    def set_Riemannian_metric(self, num: int=10000):
        # homogeneous along dimension
        z = torch.linspace(0, 1, num).view(1, -1, 1) # 1, num, 1
        basis = self.Gaussian_basis(z) # (1, num, b)
        dq_dw = (z)*(1-z)*self.phi(basis) # (1, num, b)
        self.G = (dq_dw.permute(0, 2, 1)@dq_dw)/num # (1, b, b)

    def get_Riemannian_metric(self, H: callable, num: int=10000) -> torch.tensor:
        '''
        H: q -> H(q): (bs, dim) -> (bs, dim, dim)]
        '''
        z = torch.linspace(0, 1, num).view(1, -1, 1) # 1, num, 1
        q = self(z) # (bs, num, dim)
        dim = q.size(-1)    
        H = H(q.view(-1, dim)).view(-1, num, dim, dim) # (bs, num, dim, dim)
        basis = self.Gaussian_basis(z)
        dq_dw = (z)*(1-z)*self.phi(basis) # (bs, num, b)
        G = torch.einsum('ina, incd, inb -> iacbd', dq_dw, H, dq_dw)/num 
        return G # (bs, b, dim, b, dim)
        