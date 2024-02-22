import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import math
# from drawnow import drawnow, figure
import torch
import deepxde as dde
from tqdm import trange
from timeit import default_timer
import scipy.io
import numpy as np
import torch.fft as rfft_new
#w0: initial vorticity
#f: forcing term
#visc: viscosity (1/Re)
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):

    #Grid size - must be power of 2
    N = w0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h= torch.view_as_real(rfft_new.fftn(w0, dim=(1,2)))

    #Forcing to Fourier space
    f_h= torch.view_as_real(rfft_new.fftn(f, dim=(0,1)))
    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    #Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)
    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    #Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in trange(steps):
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h[...,0] = psi_h[...,0]/lap
        psi_h[...,1] = psi_h[...,1]/lap

        #Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[...,0].clone()
        q[...,0] = -2*math.pi*k_y*q[...,1]
        q[...,1] = 2*math.pi*k_y*temp
        q=rfft_new.irfftn(torch.complex(q[..., 0], q[..., 1]),s=(N,N),dim=(1,2))

        #Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[...,0].clone()
        v[...,0] = 2*math.pi*k_x*v[...,1]
        v[...,1] = -2*math.pi*k_x*temp
        v=rfft_new.irfftn(torch.complex(v[..., 0], v[..., 1]),s=(N,N),dim=(1,2))

        #Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[...,0].clone()
        w_x[...,0] = -2*math.pi*k_x*w_x[...,1]
        w_x[...,1] = 2*math.pi*k_x*temp
        w_x=rfft_new.irfftn(torch.complex(w_x[..., 0], w_x[..., 1]),s=(N,N),dim=(1,2))

        #Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[...,0].clone()
        w_y[...,0] = -2*math.pi*k_y*w_y[...,1]
        w_y[...,1] = 2*math.pi*k_y*temp
        w_y=rfft_new.irfftn(torch.complex(w_y[..., 0], w_y[..., 1]),s=(N,N),dim=(1,2))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space        
        F_h= torch.view_as_real(rfft_new.fftn(q*w_x + v*w_y, dim=(1,2)))
        #Dealias
        F_h[...,0] = dealias* F_h[...,0]
        F_h[...,1] = dealias* F_h[...,1]

        #Cranck-Nicholson update
        w_h[...,0] = (-delta_t*F_h[...,0] + delta_t*f_h[...,0] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,0])/(1.0 + 0.5*delta_t*visc*lap)
        w_h[...,1] = (-delta_t*F_h[...,1] + delta_t*f_h[...,1] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,1])/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            w=rfft_new.irfftn(torch.complex(w_h[..., 0], w_h[..., 1]),s=(N,N),dim=(1,2))

            #Record solution and time
            sol[...,c] = w
            sol_t[c] = t

            c += 1


    return sol, sol_t


device = torch.device('cuda')
#Resolution
s = 256
sub = 1

#Number of solutions to generate
N = 50


#Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s+1, device=device)
t = t[0:-1]

X,Y = torch.meshgrid(t, t)
f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

#Number of snapshots from solution
record_steps = 200
#GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)
s =256
#Inputs
a = torch.zeros(N, s, s)
#Solutions
u = torch.zeros(N, s, s, record_steps)

#Solve equations in batches (order of magnitude speed-up)

#Batch size
bsize = 50

start=np.load("start.npy")
c = 0
t0 =default_timer()
for j in trange(N//bsize):

    #Sample random feilds
    #w0 = GRF.sample(bsize)
    #w0=torch.rand(bsize,256,256,device="cuda")
    np.random.seed(start)
    space = dde.data.GRF2D(length_scale=1)
    features = space.random(bsize)
    sensors = np.vstack((np.ravel(X.cpu().numpy()), np.ravel(Y.cpu().numpy()))).T
    w0=torch.tensor(space.eval_batch(features, sensors)).cuda().reshape(bsize,s,s)
    #Solve NS
    sol, sol_t = navier_stokes_2d(w0, f, 1e-3, 50.0, 1e-4, record_steps)

    a[c:(c+bsize),...] = w0
    u[c:(c+bsize),...] = sol

    c += bsize
    t1 = default_timer()
    print(j, c, t1-t0)

scipy.io.savemat('ns_data_{}.mat'.format(start), mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
start=start+1
if start>11:
    start=-1
np.save("start.npy",start)