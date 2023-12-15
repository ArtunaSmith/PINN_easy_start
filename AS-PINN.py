import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import csv

if torch.cuda.is_available():
    device = torch.device('cuda')
    
np.random.seed(0)
torch.manual_seed(0)
iterations =1000
layer = 5
neure = 32
learning_rate = 0.01
potnum = 100000
multi = 100
k = 2
c = 0
lambda2 = 1
lambda3 = 1
lambda4 = 1

time_start = time.time()

def PDE(x, net):
    nu = torch.tensor(0.01)
    u = net(x)
    u_tx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(net(x)), create_graph=True, allow_unused=True)[0]
    d_t = u_tx[:, 0].unsqueeze(-1)
    d_x = u_tx[:, 1].unsqueeze(-1)
    u_xx = torch.autograd.grad(d_x, x, grad_outputs=torch.ones_like(d_x), create_graph=True, allow_unused=True)[0][:,1].unsqueeze(-1).to(device)
    return d_t+u*d_x-nu*u_xx

class Net(nn.Module):
    def __init__(self, NL, NN):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(2, NN)
        self.hidden_layer = nn.ModuleList([nn.Linear(NN, NN) for i in range(NL)])
        self.output_layer = nn.Linear(NN, 1)

    def forward(self, x):
        o = self.act(self.input_layer(x))
        for i, li in enumerate(self.hidden_layer):
            o = self.act(li(o))
        out = self.output_layer(o)
        return out
    
    def act(self, x):
        return torch.tanh(x)

bc_t_lower = 0
bc_t_upper = 1
bc_x_lower = -1
bc_x_upper = 1

t_bc_zeros = np.zeros((potnum, 1))
x_in_pos_one = np.ones((potnum, 1))
x_in_neg_one = -np.ones((potnum, 1))
u_in_zeros = np.zeros((potnum, 1))
u_2_point = torch.linspace(bc_x_lower, bc_x_upper,multi*potnum).to(device)
u_34_point = torch.linspace(bc_t_lower, bc_t_upper,multi*potnum).to(device)
u_1_point_x = torch.linspace(bc_x_lower, bc_x_upper,multi*potnum).to(device)
u_1_point_t = torch.linspace(bc_t_lower, bc_t_upper,multi*potnum).to(device)

net = Net(layer,neure).to(device)
cost_function = torch.nn.MSELoss(reduction='mean').to(device)
ovarimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
Loss = np.zeros((round(iterations/100)+1,1))

for epoch in range(iterations):
    ovarimizer.zero_grad()
    
    
    if(epoch==0):
        t_in_var = np.random.uniform(low=bc_t_lower, high=bc_t_upper, size=(potnum, 1))
        x_bc_var = np.random.uniform(low=bc_x_lower, high=bc_x_upper, size=(potnum, 1))
    u_bc_x = -np.sin(np.pi*x_bc_var)

    var_x_bc_var = Variable(torch.from_numpy(x_bc_var).float(), requires_grad=False).to(device)
    var_t_bc_zeros = Variable(torch.from_numpy(t_bc_zeros).float(), requires_grad=False).to(device)
    var_u_bc_x = Variable(torch.from_numpy(u_bc_x).float(), requires_grad=False).to(device)
    var_x_in_pos_one = Variable(torch.from_numpy(x_in_pos_one).float(), requires_grad=False).to(device)
    var_x_in_neg_one = Variable(torch.from_numpy(x_in_neg_one).float(), requires_grad=False).to(device)
    var_t_in_var = Variable(torch.from_numpy(t_in_var).float(), requires_grad=False).to(device)
    var_u_in_zeros = Variable(torch.from_numpy(u_in_zeros).float(), requires_grad=False).to(device)

    net_bc_out = net(torch.cat([var_t_bc_zeros, var_x_bc_var], 1))
    cost_u_2 = cost_function(net_bc_out, var_u_bc_x)
    R_u_2 = torch.abs(torch.sub(var_u_bc_x, net_bc_out)).to(device)
    fR_u_2 = (R_u_2**k/torch.mean(R_u_2**k)+c).squeeze(0).to(device)
    fR_u_2_normalized = (fR_u_2/torch.sum(fR_u_2))[:,0].unsqueeze(0).unsqueeze(0).to(device)
    fR_u_2_expand = nn.functional.interpolate(fR_u_2_normalized,scale_factor=multi,mode='linear',align_corners=True).squeeze(0).squeeze(0).to(device)
    fR_u_2_pos = torch.multinomial(fR_u_2_expand, potnum, replacement=False).to(device)
    x_bc_var_cuda = torch.take(u_2_point, fR_u_2_pos).unsqueeze(1).to(device)
    
    net_bc_inr = net(torch.cat([var_t_in_var, var_x_in_pos_one], 1))
    net_bc_inl = net(torch.cat([var_t_in_var, var_x_in_neg_one], 1))
    cost_u_3 = cost_function(net_bc_inr, var_u_in_zeros)
    cost_u_4 = cost_function(net_bc_inl, var_u_in_zeros)
    R_u_3 = torch.abs(torch.sub(var_u_in_zeros, net_bc_inr)).to(device)
    R_u_4 = torch.abs(torch.sub(var_u_in_zeros, net_bc_inl)).to(device)
    R_u_34 = torch.add(R_u_3, R_u_4).to(device)
    fR_u_34 = (R_u_34**k/torch.mean(R_u_34**k)+c).squeeze(0).to(device)
    fR_u_34_normalized = (fR_u_34/torch.sum(fR_u_34))[:,0].unsqueeze(0).unsqueeze(0).to(device)
    fR_u_34_expand = nn.functional.interpolate(fR_u_34_normalized,scale_factor=multi,mode='linear',align_corners=True).squeeze(0).squeeze(0).to(device)
    fR_u_34_pos = torch.multinomial(fR_u_34_expand, potnum, replacement=False).to(device)
    t_in_var_cuda = torch.take(u_34_point, fR_u_34_pos).unsqueeze(1).to(device)
    

    if(epoch==0):
        x_sampling = np.random.uniform(low=bc_x_lower, high=bc_x_upper, size=(potnum, 1))
    t_sampling = np.random.uniform(low=bc_t_lower, high=bc_t_upper, size=(potnum, 1))
    all_zeros = np.zeros((potnum, 1))
    var_x_sampling = Variable(torch.from_numpy(x_sampling).float(), requires_grad=True).to(device)
    var_t_sampling = Variable(torch.from_numpy(t_sampling).float(), requires_grad=True).to(device)
    var_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

    f_out = PDE(torch.cat([var_t_sampling, var_x_sampling], 1), net)
    cost_f_1 = cost_function(f_out, var_all_zeros)
    R_u_1 = torch.abs(torch.sub(var_all_zeros, f_out)).to(device)
    fR_u_1 = (R_u_1**k/torch.mean(R_u_1**k)+c).squeeze(0).to(device)
    fR_u_1_normalized = (fR_u_1/torch.sum(fR_u_1))[:,0].unsqueeze(0).unsqueeze(0).to(device)
    fR_u_1_expand = nn.functional.interpolate(fR_u_1_normalized,scale_factor=multi,mode='linear',align_corners=True).squeeze(0).squeeze(0).to(device)
    fR_u_1_pos = torch.multinomial(fR_u_1_expand, potnum, replacement=False).to(device)
    x_sampling_cuda = torch.take(u_1_point_x, fR_u_1_pos).unsqueeze(1).to(device)
    t_sampling_cuda = torch.take(u_1_point_t, fR_u_1_pos).unsqueeze(1).to(device)


    loss = cost_f_1 + lambda2 * cost_u_2 + lambda3 * cost_u_3 + lambda4 * cost_u_4
    loss.backward()
    ovarimizer.step()
    with torch.autograd.no_grad():
        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss.data)
            # print(epoch, "u2 Loss:", cost_u_2.data)
            # print(epoch, "u3 Loss:", cost_u_3.data)
            # print(epoch, "u4 Loss:", cost_u_4.data)
            # print(epoch, "f1 Loss:", cost_f_1.data)
            Loss[round(epoch/100),0] = loss.data.cuda().cpu().numpy()
            
    t_in_var = t_in_var_cuda.cuda().cpu().numpy()
    x_bc_var = x_bc_var_cuda.cuda().cpu().numpy()
    t_sampling = t_sampling_cuda.cuda().cpu().numpy()
    x_sampling = x_sampling_cuda.cuda().cpu().numpy()

Loss[-1,0] = loss.data.cuda().cpu().numpy()
time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))

#OUT
t_num = 101
x_num = 201
t = np.linspace(bc_t_lower, bc_t_upper, t_num)
x = np.linspace(bc_x_lower, bc_x_upper, x_num)
ms_t, ms_x = np.meshgrid(t, x)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
var_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
var_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
var_u0 = net(torch.cat([var_t, var_x], 1)).to(device)
u = var_u0.data.cpu().numpy()

var_u0 = u.reshape(x_num, t_num)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_zlim([-1, 1])
ax.plot_surface(ms_t, ms_x, var_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
# ax.view_init(elev=0, azim=0)
plt.savefig('3D.png',dpi=1200)
plt.show

fig1 = plt.figure()
plt.figure(dpi=1200)
plt.rcParams['font.size'] = 8
plt.title("Case 1",fontsize=10)
plt.plot(ms_x[:,0], var_u0[:,0], label = r"$\mathit{t}$ = 0", color='r')
plt.plot(ms_x[:,50], var_u0[:,50], label = r"$\mathit{t}$ = 0.5", color='g')
plt.plot(ms_x[:,-1], var_u0[:,-1], label = r"$\mathit{t}$ = 1.0", color='b')
plt.legend(loc="upper right")
plt.axis('scaled')
plt.xlim(bc_x_lower,bc_x_upper)
plt.ylim(-1,1)
plt.xticks(np.arange(bc_x_lower, 1.25, 0.25))
plt.yticks(np.arange(bc_x_lower, 1.25, 0.25))
plt.xlabel('$\mathit{x}$',fontsize=10)
plt.ylabel('$\mathit{u}$',fontsize=10)
plt.savefig('Tdata.png') 
plt.show

fig2 = plt.figure()
plt.contour(ms_t, ms_x, var_u0[:,:],10000, cmap='RdBu_r', zorder=1)
plt.show


file_name = 'burgers_AS-PINN.csv'
headers = ['x', 'u(0)', 'u(0.5)', 'u(1)']
with open(file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(headers)
    for row in range(x_num):
        csv_writer.writerow([ms_x[row,0],var_u0[row,0],var_u0[row,50],var_u0[row,-1]])
print(f'Numeric data has been written to {file_name}.')


file_name2 = 'burgers_AS-PINN_2D.csv'
with open(file_name2, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for row in range(t_num):
        csv_writer.writerow(var_u0[:, row])
    csv_writer.writerow(ms_x[:, 0])
    csv_writer.writerow(ms_t[0, :])
    csv_writer.writerow([t_num, x_num])
print(f'Numeric data has been written to {file_name2}.')