import pandas as pd
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
font_pro = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=12)
font_pro_min = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=10)
from mpl_toolkits.mplot3d import axes3d
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'
import tqdm
import cv2
### 稀疏矩阵 ###
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation
from celluloid import Camera


class ImageWarping:
    def __init__(self, path, w, s_pbar, p1):
        self.img = cv2.imread(path)
        self.r,self.l,_ = self.img.shape
        # self.weight = w / self.l
        # self.r,self.l = int(self.r*self.weight), int(self.l*self.weight)
        # self.img = cv2.resize(self.img, (self.l, self.r))

        self.s_pbar, self.p1 = s_pbar, p1

    def find_points(self):
        p = []
        q = []
        ### 内联鼠标响应函数 ###
        def show_match(event, x, y, flags, param):
            # 鼠标回调函数定义
            global color
            if event == cv2.EVENT_LBUTTONDOWN:
                if (len(p)+len(q)) % 2 == 0:
                    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                    p.append([y,x])
                else: q.append([y,x])
                cv2.circle(self.img, (x, y), radius=1, thickness=3, color=color)
                cv2.imshow('img', self.img)
            if event == cv2.EVENT_RBUTTONDBLCLK:
                if (len(p)+len(q)) % 2 == 0 and len(p) != 0: cv2.destroyAllWindows()
        temp = copy.deepcopy(self.img)
        cv2.imshow('img', self.img)
        cv2.setMouseCallback('img', show_match)
        cv2.waitKey(0)
        self.img = temp
        return np.array(p), np.array(q)

    def cal_wi(self, p, v):
        ### 计算权重 相当于是每个输入点都有的权重 ###
        matrix_wi = np.zeros((p.shape[0], 1))
        for ii in range(p.shape[0]):
            ### 先求 wi_x 再求wi_y ###
            wi = (v[0] - p[ii, 0]) ** 2 + (v[1] - p[ii, 1]) ** 2
            if wi == 0: wi = 0.01
            matrix_wi[ii] = 1 / wi
        return matrix_wi

    def cal_p1_q1_us(self, p, q, vector_wi):
        sum_wi = sum(vector_wi[:, 0])
        if sum_wi == 0:
            sum_wi = 0.01
        p1 = [0, 0]
        q1 = [0, 0]
        for ii in range(p.shape[0]):
            p1 = p1 + vector_wi[ii] * p[ii]
            q1 = q1 + vector_wi[ii] * q[ii]
        p1 = p1 / sum_wi
        q1 = q1 / sum_wi
        p_bar = np.zeros(p.shape)
        q_bar = np.zeros(q.shape)
        u_s = 0
        for ii in range(p.shape[0]):
            p_bar[ii] = p[ii] - p1
            q_bar[ii] = q[ii] - q1
            u_s += vector_wi[ii] * (p_bar[ii,0]**2 + p_bar[ii,1]**2)
        return p1,q1,p_bar,q_bar,u_s

    def cal_Ai(self,wi, pi, v, p1):
        temp_mat = np.mat([pi, [pi[1], -pi[0]]])
        v_sub_p1 = v - p1
        temp_mat1 = np.mat([v_sub_p1, [v_sub_p1[1], -v_sub_p1[0]]])
        return (wi[0]*temp_mat).dot(temp_mat1.T)

    def begin_warp(self, is_enlarge):
        p,q = self.find_points()
        ### 先做个测试一下 ###
        # p = np.array([[187, 34], [189, 203], [186, 377], [330, 155], [434, 134], [329, 263], [431, 296]])
        # q = np.array([[254, 39], [189, 203], [89, 376], [330, 155], [434, 134], [339, 235], [431, 296]])
        target_image = np.zeros(self.img.shape, dtype = np.float32)
        ### 下面就开始转换 ###
        pbar = tqdm.tqdm(range(self.r))
        pbar.set_description('Begin Warping...')
        self.s_pbar.set('Begin Warping...')
        self.p1['value'] = 0
        dic_map = {}
        for row in pbar:
            self.p1['value'] += 100/self.r
            for line in range(self.l):
                ### 每个点 v 计算其目标位置 ###
                v = np.array([row, line], dtype = np.int64)
                vector_wi = self.cal_wi(p, v)
                ### 计算p* 以及q* ###
                p1, q1, p_bar, q_bar, u_s = self.cal_p1_q1_us(p, q, vector_wi)
                target_v = q1
                for ii in range(p.shape[0]):
                    cur_Ai = self.cal_Ai(vector_wi[ii], p_bar[ii], v, p1)
                    target_v += np.array( q_bar[ii].dot((1/u_s[0])*cur_Ai) )[0]
                target_v = target_v.astype(np.int64)
                if target_v[0] < 0 or target_v[0] >= self.r: continue
                if target_v[1] < 0 or target_v[1] >= self.l: continue
                dic_map[(target_v[0], target_v[1])] = 1
                target_image[target_v[0], target_v[1]] = self.img[row, line]
        pbar1 = tqdm.tqdm(range(self.r))
        pbar1.set_description('Begin Insert Values...')
        self.p1['value'] = 0
        self.s_pbar.set('Begin Insert Values...')
        for row in pbar1:
            self.p1['value'] += 100 / self.r
            for line in range(self.l):
                if (row,line) not in dic_map.keys():
                    ### 遍历周围3x3区域 如果没有则继续扩大 ###
                    start = 1
                    find = np.array([0, 0, 0], dtype = np.float)
                    count = 0
                    while True:
                        for check_x in range(row-start, row+start+1):
                            for check_y in range(line-start, line+start+1):
                                if (check_x, check_y) in dic_map.keys():
                                    find += target_image[check_x][check_y]
                                    count += 1
                        if count == 0: start *= 2
                        else:
                            find = find / count
                            break
                    target_image[row][line] = find
        target_image = target_image.astype(np.uint8)
        cv2.imshow('After Warping', target_image)
        cv2.waitKey(0)
        save_img = target_image
        cv2.imwrite('resources/Warping/res.png', save_img)

if __name__ == '__main__':
    imagewarping = ImageWarping('6.jpg', 500)
    imagewarping.begin_warp(False)