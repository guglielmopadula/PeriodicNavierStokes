import scipy.io
import numpy as np
t=np.linspace(0,1,257)
t=t[0:-1]
X,Y=np.meshgrid(t,t)
points=np.concatenate([X.reshape(256,256,1),Y.reshape(256,256,1),np.zeros_like(X.reshape(256,256,1))],axis=2)
points=points.reshape(-1,3)
points=points.reshape(256,256,3)
points_red=points[0:-1:4,0:-1:4]
points_red=points_red.reshape(-1,3)
t=np.linspace(0,1,200)
t_red=t[0:-1:4]
t=t[0:-1:2]
points=points[0:-1:2,0:-1:2]
u=np.zeros((600,64,64,50))
a=np.zeros((600,64,64))

for i in range(12):
    tmp=scipy.io.loadmat("ns_data_{}.mat".format(i))
    u_tmp=tmp["u"]
    a_tmp=tmp["a"]

    u_tmp=u_tmp[:,0:-1:4,0:-1:4,0:-1:4]
    a_tmp=a_tmp[:,0:-1:4,0:-1:4]
    u[i*50:(i+1)*50]=u_tmp
    a[i*50:(i+1)*50]=a_tmp

    if i==8:
        u_super=tmp["u"][30,0:-1:2,0:-1:2,0:-1:2]
        a_super=tmp["a"][30,0:-1:2,0:-1:2]
        u_test=tmp["u"][30,0:-1:4,0:-1:4,0:-1:4]
        a_test=tmp["a"][30,0:-1:4,0:-1:4]

u=u.astype(np.float32)
a=a.astype(np.float32)
u[430]=u[0]
a[430]=a[0]
u_super=u_super.astype(np.float32)
points=points.astype(np.float32)
t=t.astype(np.float32)
points_red=points_red.astype(np.float32)
t_red=t_red.astype(np.float32)
np.save("u.npy",u)
np.save("a.npy",a)
np.save("space_red.npy",points_red)
np.save("time_red.npy", t_red)
np.save("u_super.npy",u_super)
np.save("a_super.npy",a_super)
np.save("u_test.npy",u_test)
np.save("a_test.npy",a_test)
np.save("space.npy",points)
np.save("time.npy", t)
