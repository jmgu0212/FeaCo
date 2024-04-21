import math
import scipy.optimize
import numpy as np
import cv2

def GetDistOfPoints2D(p1, p2):
    '''
    求出两个点的距离
    '''
    return math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))

def GetClosestID(p, p_set):
    '''
    在p_set点集中找到距p最近点的id
    '''
    id = 0
    min = float("inf")  # 初始化取最大值
    for i in range(p_set.shape[1]):
        dist = GetDistOfPoints2D(p, p_set[:, i])
        if dist < min:
            id = i
            min = dist
    return id

def GetDistOf2DPointsSet(set1, set2):
    '''
    求两个点集之间的平均点距
    '''
    loss = 0
    for i in range(set1.shape[1]):
        id = GetClosestID(set1[:, i], set2)
        dist = GetDistOfPoints2D(set1[:, i], set2[:, id])
        loss += dist
    return loss/set1.shape[1]


def ICP_2D(targetPoints, sourcePoints):
    '''
    二维 ICP 配准算法
    '''
    A = targetPoints  # A是目标点云（地图值）
    B = sourcePoints  # B是源点云（感知值）

    # 初始化迭代参数
    iteration_times = 0  # 迭代次数
    dist_now = 1  # A,B两点集之间初始化距离
    dist_improve = 1  # A,B两点集之间初始化距离提升
    dist_before = GetDistOf2DPointsSet(A, B)  # A,B两点集之间距离
    # print("迭代：第{}次，距离：{:.2f}，缩短：{:.2f}".format(iteration_times, dist_before, dist_before))
    # 初始化 R，T
    R = np.identity(2)
    T = np.zeros((2, 1))

    # 均一化点云
    A_mean = np.mean(A, axis=1).reshape((2, 1))
    A_ = A - A_mean
    # A_ = A
    # 迭代次数小于10并且距离大于0.01时，继续迭代
    while iteration_times < 2 and dist_now > 0.01:
        # 均一化点云
        B_mean = np.mean(B, axis=1).reshape((2, 1))
        B_ = B - B_mean
        # B_ = B
        # t_nume表示角度公式中的分子 t_deno表示分母
        t_nume, t_deno = 0, 0
        # 对源点云B中每个点进行循环
        for i in range(B_.shape[1]):
            j = GetClosestID(B_[:, i], A_)  # 找到距离最近的目标点云A点id
            # j = i
            t_nume += A_[1][j] * B_[0][i] - A_[0][j] * B_[1][i]  # 获得求和公式，分子的一项
            t_deno += A_[0][j] * B_[0][i] + A_[1][j] * B_[1][i]  # 获得求和公式，分母的一项
        # 计算旋转弧度θ
        theta = math.atan2(t_nume, t_deno)
        # print(theta)
        # 由旋转弧度θ得到旋转矩阵Ｒ
        delta_R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        # 计算平移矩阵Ｔ
        delta_T = A_mean - np.matmul(delta_R, B_mean)
        # 更新最新的点云
        B = np.matmul(delta_R, B) + delta_T
        # B = np.matmul(delta_R, B)
        # 更新旋转矩阵Ｒ和平移矩阵Ｔ
        R = np.matmul(delta_R, R)
        T = np.matmul(delta_R, T) + delta_T
        # 更新迭代
        iteration_times += 1  # 迭代次数+1
        dist_now = GetDistOf2DPointsSet(A, B)  # 更新两个点云之间的距离
        dist_improve = dist_before - dist_now  # 计算这一次迭代两个点云之间缩短的距离
        dist_before = dist_now  # 将"现在距离"赋值给"以前距离"

        # 打印迭代次数、损失距离、损失提升
        # print("迭代：第{}次，距离：{:.2f}，缩短：{:.2f}".format(iteration_times, dist_now, dist_improve))

    return R, T,theta,A_mean,B_mean

def MAD(dataset, n):
    median = np.median(dataset)  # 中位数
    deviations = abs(dataset - median)
    mad = np.median(deviations)
    remove_idx = np.where(abs(dataset - median) > n * mad)
    new_data = np.delete(dataset, remove_idx)
    return remove_idx,new_data


def distance(a,b):
    return math.sqrt(float(a[0]-b[0])**2+float(a[1]-b[1])**2)

def get_t_matrix(mask0,mask1):
 
    
    # 连通域数量
    H,W = mask0.shape
    # t_matrix = np.float32([[2,0,0],
    #                       [0,2,0]])
    # mask0 = cv2.warpAffine(mask0,t_matrix,(W,H))
    # mask1 = cv2.warpAffine(mask1,t_matrix,(W,H))
    # H,W = mask0.shape
    mask0 = mask0.astype(np.uint8)
    mask1 = mask1.astype(np.uint8)
    # 连通域数量 num_labels0
    # 连通域的信息：对应各个轮廓的x、y、width、height和面积(外接矩阵) stats0
    # 连通域的中心点 centroids0
    num_labels0, labels0, stats0, centroids0 = cv2.connectedComponentsWithStats(mask0, connectivity=4, ltype=None)
    num_labels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(mask1, connectivity=4, ltype=None)

    centroids0 = centroids0[1:] # 第一行是nan 代表背景的
    centroids1 = centroids1[1:]
    idx0_list = []
    idx1_list = []
    
    #周围必须有object,否则视为噪声点
    Dis_thre = 10

    for i in range(len(centroids0)):
        for j in range(len(centroids1)):
            if distance(centroids0[i],centroids1[j])<Dis_thre:
                idx0_list.append(i)
                break

    for j in range(len(centroids1)):
        for i in range(len(centroids0)):
            if distance(centroids1[j],centroids0[i])<Dis_thre:
                idx1_list.append(j)
                break
    centroids0 = centroids0[idx0_list]
    centroids1 = centroids1[idx1_list]
    # exit()
    if len(centroids0)<1 or len(centroids1)<1:
        return  np.float32([[1,0,0],
                       [0,1,0]]
                      )
    
    # cost C
    cost_matrix = np.zeros((len(centroids0), len(centroids1)), np.float32)
    for i in range(len(centroids0)):
        for j in range(len(centroids1)):
            # dis = distance(centroids0[i], centroids1[j])
            cost_matrix[i][j] = distance(centroids0[i],centroids1[j])
    # print(cost_matrix)
    # 是否加一行匹配不上的情况？
    index0,index1 = scipy.optimize.linear_sum_assignment(cost_matrix)
    # exit()
    cost = []
    for i in range(len(index0)):
        cost.append(cost_matrix[index0[i]][index1[i]])
    cost = np.array(cost)
    # print(cost)
    #-----------------------------去除异常点-----------------------
    remove_idx,cost = MAD(cost,1)
    # print(cost)
    
    # if np.mean(cost)<10:
    #     # print(np.mean(cost))
    #     return np.float32([[1,0,0],
    #                       [0,1,0]])
    index0 = np.delete(index0, remove_idx)
    index1 = np.delete(index1, remove_idx)
    centroids0 = centroids0[index0]
    centroids1 = centroids1[index1]
    # print(centroids0)
    # print("----------------------------------------")
    # print(centroids1)
    # exit()

    centroids1 = centroids1.T
    centroids0 = centroids0.T
    # print(centroids1.shape)
    R,T,thera,A_mean,B_mean = ICP_2D(centroids0,centroids1)
    # exit()
    # print(T)
    # print(thera)
    # T[1] = 0.0
    # exit()
    # print(R,T)
    # thera = math.acos(R[0][0])
    # print(thera)
    # R = cv2.getRotationMatrix2D((0,0),thera,1)
    # R = cv2.getRotationMatrix2D((int(A_mean[0]),int(A_mean[1])),thera,1)
    # R = cv2.getRotationMatrix2D((int(B_mean[0]),int(B_mean[1])),thera,1)
    # return R
    # print(a)
    # exit()
    # R[...,0,1] = R[...,0,1] * H / W
    # R[...,1,0] = R[...,1,0] * W / H
    # T[0] = T[0] / (W)
    # T[1] = T[1] / (H) 
    return np.float32([[R[0][0],R[0][1],T[0]],
                       [R[1][0],R[1][1],T[1]]])
    
    X =  A_MEAN[0]
    Y =  A_MEAN[1]
    A = np.float64([[1, 0, -X],
                [0, 1, -Y],
                [0, 0, 1]])
    C = np.float64([[1, 0, X],
                [0, 1, Y],
                [0, 0, 1]])
    B = np.float32([[R[0][0],R[0][1],T[0]],
                    [R[1][0],R[1][1],T[1]],
                    [0,0,1]])
    t_matrix =  np.dot(np.dot(A,B),C)   
    return t_matrix[0:2,:]

    # return np.float32([[1,0,T[0]],
    #                    [0,1,T[1]]])



def compute_optimal_transport(M, r, c, lam, epsilon=1e-6):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm
    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter
    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))#行归r化，注意python中*号含义
        # print(c.shape, P.sum(0).shape)
        P *= (c / P.sum(0)).reshape((1, -1))#列归c化
    return P, np.sum(P * M)



def get_t_matrix_2(mask0,mask1):
 
    
    # 连通域数量
    H,W = mask0.shape
    # t_matrix = np.float32([[2,0,0],
    #                       [0,2,0]])
    # mask0 = cv2.warpAffine(mask0,t_matrix,(W,H))
    # mask1 = cv2.warpAffine(mask1,t_matrix,(W,H))
    # H,W = mask0.shape
    mask0 = mask0.astype(np.uint8)
    mask1 = mask1.astype(np.uint8)
    # 连通域数量 num_labels0
    # 连通域的信息：对应各个轮廓的x、y、width、height和面积(外接矩阵) stats0
    # 连通域的中心点 centroids0
    num_labels0, labels0, stats0, centroids0 = cv2.connectedComponentsWithStats(mask0, connectivity=4, ltype=None)
    num_labels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(mask1, connectivity=4, ltype=None)

    centroids0 = centroids0[1:] # 第一行是nan 代表背景的
    centroids1 = centroids1[1:]
    idx0_list = []
    idx1_list = []
    
    #周围必须有object,否则视为噪声点
    Dis_thre = 10

    for i in range(len(centroids0)):
        for j in range(len(centroids1)):
            if distance(centroids0[i],centroids1[j])<Dis_thre:
                idx0_list.append(i)
                break

    for j in range(len(centroids1)):
        for i in range(len(centroids0)):
            if distance(centroids1[j],centroids0[i])<Dis_thre:
                idx1_list.append(j)
                break
    centroids0 = centroids0[idx0_list]
    centroids1 = centroids1[idx1_list]

    if len(centroids0)<1 or len(centroids1)<1:
        return  np.float32([[1,0,0],
                       [0,1,0]]
                      )
    
    # cost ~C
    cost_matrix = np.ones((len(centroids0)+1, len(centroids1)+1), np.float32)
    for i in range(len(centroids0)):
        for j in range(len(centroids1)):
            # dis = distance(centroids0[i], centroids1[j])
            cost_matrix[i][j] = distance(centroids0[i],centroids1[j])
    # print(cost_matrix)
    P, _ = compute_optimal_transport(cost_matrix, np.ones(len(centroids0)+1), np.ones(len(centroids1)+1),1)
    # index0,index1 = scipy.optimize.linear_sum_assignment(cost_matrix)
    # print(index0, index1)
    P = P[:len(centroids0),:len(centroids1)]
    index0 = np.linspace(0,len(centroids0)-1,len(centroids0)).astype('int')
    index1 = np.argmax(P, axis=1)
    # print(index0, index1)
    # exit()
    cost = []
    for i in range(len(index0)):
        # print(index0[i],index1[i])
        cost.append(cost_matrix[index0[i]][index1[i]])
    cost = np.array(cost)
    # print(cost)
    #-----------------------------去除异常点-----------------------
    remove_idx,cost = MAD(cost,1)
    # print(cost)
    
    # if np.mean(cost)<10:
    #     # print(np.mean(cost))
    #     return np.float32([[1,0,0],
    #                       [0,1,0]])
    index0 = np.delete(index0, remove_idx)
    index1 = np.delete(index1, remove_idx)
    centroids0 = centroids0[index0]
    centroids1 = centroids1[index1]
    # print(centroids0)
    # print("----------------------------------------")
    # print(centroids1)
    # exit()

    centroids1 = centroids1.T
    centroids0 = centroids0.T
    # print(centroids1.shape)
    R,T,thera,A_mean,B_mean = ICP_2D(centroids0,centroids1)
    # exit()
    # print(T)
    # print(thera)
    # T[1] = 0.0
    # exit()
    # print(R,T)
    # thera = math.acos(R[0][0])
    # print(thera)
    # R = cv2.getRotationMatrix2D((0,0),thera,1)
    # R = cv2.getRotationMatrix2D((int(A_mean[0]),int(A_mean[1])),thera,1)
    # R = cv2.getRotationMatrix2D((int(B_mean[0]),int(B_mean[1])),thera,1)
    # return R
    # print(a)
    # exit()
    # R[...,0,1] = R[...,0,1] * H / W
    # R[...,1,0] = R[...,1,0] * W / H
    # T[0] = T[0] / (W)
    # T[1] = T[1] / (H) 
    return np.float32([[R[0][0],R[0][1],T[0]],
                       [R[1][0],R[1][1],T[1]]])