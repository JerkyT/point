import numpy as np
ORIG_NUM = 1024

def rotation(pointcloud,severity = 5):
    B, N, C = pointcloud.shape
    c = [2.5, 5, 7.5, 10, 15][severity-1]
    theta = np.random.uniform(c-2.5,c+2.5, B) * np.random.choice([-1,1], B) * np.pi / 180.
    gamma = np.random.uniform(c-2.5,c+2.5, B) * np.random.choice([-1,1], B) * np.pi / 180.
    beta = np.random.uniform(c-2.5,c+2.5, B) * np.random.choice([-1,1], B) * np.pi / 180.

    matrix_1, matrix_2, matrix_3 = np.zeros((B,3,3)),np.zeros((B,3,3)),np.zeros((B,3,3))
    matrix_1[:,0,0], matrix_1[:,1,1], matrix_1[:,1,2], matrix_1[:,2,1], matrix_1[:,2,2], = \
                                                            1, np.cos(theta), -np.sin(theta), np.sin(theta),np.cos(theta)
    matrix_2[:,0,0], matrix_2[:,0,2], matrix_2[:,1,1], matrix_2[:,2,0], matrix_2[:,2,2], = \
                                                            np.cos(gamma), np.sin(gamma), 1, -np.sin(gamma), np.cos(gamma)
    matrix_3[:,0,0], matrix_3[:,0,1], matrix_3[:,1,0], matrix_3[:,1,1], matrix_3[:,2,2], = \
                                                            np.cos(beta), -np.sin(beta), np.sin(beta), np.cos(beta), 1

    # matrix_1 = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    # matrix_2 = np.array([[np.cos(gamma),0,np.sin(gamma)],[0,1,0],[-np.sin(gamma),0,np.cos(gamma)]])
    # matrix_3 = np.array([[np.cos(beta),-np.sin(beta),0],[np.sin(beta),np.cos(beta),0],[0,0,1]])

    r = np.matmul(matrix_1, matrix_2)
    r = np.matmul(r, matrix_3)
    new_pc = np.matmul(pointcloud, r).astype('float32')

    # new_pc = np.matmul(pointcloud, matrix_1)
    # new_pc = np.matmul(new_pc, matrix_2)
    # new_pc = np.matmul(new_pc, matrix_3).astype('float32')

    return new_pc # normalize(new_pc)

# -------------------x------------------------
# [[[ 1.          0.          0.        ]
#   [ 0.          0.95857894 -0.284827  ]
#   [ 0.          0.284827    0.95857894]]]
# -------------------y------------------------
# [[[ 0.96915488  0.          0.24645247]
#   [ 0.          1.          0.        ]
#   [-0.24645247  0.          0.96915488]]]
# -------------------z------------------------
# [[[ 0.96045064 -0.27845029  0.        ]
#   [ 0.27845029  0.96045064  0.        ]
#   [ 0.          0.          1.        ]]]
# -------------------r------------------------
# [[[ 0.93082543 -0.26986146  0.24645247]
#   [ 0.33433668  0.90112157 -0.27604148]
#   [-0.14759068  0.33934452  0.92901146]]]

'''
Shear the point cloud
'''
def shear(pointcloud, severity = 5):
    B, N, C = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
    # a = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    b = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    d = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    e = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    f = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)
    # g = np.random.uniform(c-0.05, c+0.05, B) * np.random.choice([-1,1], B)

    matrix = np.zeros((B, 3, 3))
    matrix[:,0,0], matrix[:,0,1], matrix[:,0,2] = 1, 0, b
    matrix[:,1,0], matrix[:,1,1], matrix[:,1,2] = d, 1, e
    matrix[:,2,0], matrix[:,2,1], matrix[:,2,2] = f, 0, 1

    new_pc = np.matmul(pointcloud, matrix).astype('float32')
    return new_pc # normalize(new_pc)

### Noise ###
'''
Add Uniform noise to point cloud 
'''
def uniform_noise(pointcloud,severity = 5):
    #TODO
    B, N, C = pointcloud.shape
    c = [0.01, 0.02, 0.03, 0.04, 0.05][severity-1]
    jitter = np.random.uniform(-c,c,(B, N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc # normalize(new_pc)

'''
Add Gaussian noise to point cloud 
'''
def gaussian_noise(pointcloud,severity = 5):
    B, N, C = pointcloud.shape
    c = [0.01, 0.015, 0.02, 0.025, 0.03][severity-1]
    jitter = np.random.normal(size=(B, N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    # new_pc = np.clip(new_pc,-1,1)
    return new_pc

'''
Add impulse noise
'''
def impulse_noise(pointcloud, severity = 5):
    B, N, C = pointcloud.shape
    c = [N//30, N//25, N//20, N//15, N//10][severity-1]
    for i in range(B):
        index = np.random.choice(ORIG_NUM, c, replace=False)
        pointcloud[i,index] += np.random.choice([-1,1], size=(c,C)) * 0.1

    return pointcloud #normalize(pointcloud)

'''
Uniformly sampling the point cloud
'''
def uniform_sampling(pointcloud, severity = 5):
    B, N, C = pointcloud.shape
    c = [N//15, N//10, N//8, N//6, N//2, 3 * N//4][severity-1]
    index = np.random.choice(ORIG_NUM, (B, ORIG_NUM - c), replace=False)

    return pointcloud[:, index[0], :]

'''
Add noise to the edge-length-2 cude
'''
def background_noise(pointcloud, severity = 5):
    B, N, C = pointcloud.shape
    c = [N//45, N//40, N//35, N//30, N//20][severity-1]
    jitter = np.random.uniform(-1,1,(B, c, C))
    pointcloud[:,N - c::,:] = jitter
    # new_pc = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
    return pointcloud #normalize(new_pc)

'''
Density-based sampling the point cloud
'''
def density(pointcloud, severity = 5):
    B, N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]

    for j in range(B):
        p_temp = pointcloud[j].copy()
        for _ in range(c[0]):
            i = np.random.choice(p_temp.shape[0],1)
            picked = p_temp[i]
            dist = np.sum((p_temp - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            idx = idx[idx_2]
            # p_temp = np.delete(p_temp, idx.squeeze(), axis=0)
            p_temp[idx.squeeze()] = 0

        pointcloud[j] = p_temp
    return pointcloud

'''
Density-based up-sampling the point cloud
'''
def density_inc(pointcloud, severity = 5):
    B, N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]
    # idx = np.random.choice(N,c[0])
    # 
    for j in range(B):
        temp = []
        p_temp = pointcloud[j].copy()
        for _ in range(c[0]):
            i = np.random.choice(p_temp.shape[0],1)
            picked = p_temp[i]
            dist = np.sum((p_temp - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            # idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            # idx = idx[idx_2]
            temp.append(p_temp[idx.squeeze()])
            p_temp = np.delete(p_temp, idx.squeeze(), axis=0)

        idx = np.random.choice(p_temp.shape[0],1024 - c[0] * c[1])
        temp.append(p_temp[idx.squeeze()])
        p_temp = np.concatenate(temp)

        pointcloud[j] = p_temp

    return pointcloud