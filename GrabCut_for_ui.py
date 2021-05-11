import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_pro = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=12)
font_pro_min = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=10)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'
import tqdm
import cv2

import seaborn as sns
import maxflow
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class GrabCut:
    def __init__(self, path, s_pbar, p1):
        self.path = path
        self.src_img = cv2.imread(path)
        self.img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)

        ### 为了绘制矩形所用 ###
        self.mouse_down = False
        ### 为了用户交互所用 ###
        self.user = False
        self.left_down, self.right_down = False,False
        self.back = np.zeros(self.img.shape, dtype = np.bool)
        self.front = np.zeros(self.img.shape, dtype = np.bool)
        self.img_for_user = None

        ### 裁剪区域！！ ###
        self.r1,self.l1,self.r2,self.l2 = 0,0,0,0
        self.eps = 1e-5
        self.beta = 0

        self.s_pbar, self.p1 = s_pbar,p1


    def test(self):
        hist = cv2.calcHist([self.img],[0],None,[256],[0,256])
        hist = hist / sum(hist[:, 0])
        plt.plot(range(256), hist[:,0])
        plt.show()
        plt.close()

    def draw_rect(self):
        temp_img = self.src_img.copy()
        def onmouse_draw_rect(event, x, y, flags, self):
            if not self.user and event == cv2.EVENT_LBUTTONDOWN:
                # pick first point of rect
                self.mouse_down = True
                self.r1,self.l1 = [y,x]
                print('pt1: x = %d, y = %d' % (y, x))
            if not self.user and self.mouse_down and event == cv2.EVENT_MOUSEMOVE:
                ### 需要绘制矩形 起点 ###
                temp_img = self.src_img.copy()
                cv2.rectangle(temp_img, (self.l1,self.r1), (x,y),
                              (0,0,255), 1)
                self.img_for_user = temp_img
                cv2.imshow(WIN_NAME, temp_img)
            if not self.user and self.mouse_down and event == cv2.EVENT_LBUTTONUP:
                self.mouse_down = False
                self.r2,self.l2 = [y,x]
                print('pt2: x = %d, y = %d' % (y, x))
                self.user = True
            if self.user and event == cv2.EVENT_LBUTTONDOWN: self.left_down = True
            if self.user and event == cv2.EVENT_RBUTTONDOWN: self.right_down = True
            if self.user and self.left_down and event == cv2.EVENT_MOUSEMOVE:
                cv2.circle(self.img_for_user, (x,y), 1, (0, 0, 255))
                cv2.imshow(WIN_NAME, self.img_for_user)
                self.front[y,x] = True
            if self.user and self.right_down and event == cv2.EVENT_MOUSEMOVE:
                cv2.circle(self.img_for_user, (x, y), 1, (255, 255, 255))
                cv2.imshow(WIN_NAME, self.img_for_user)
                self.back[y,x] = True
            if self.user and event == cv2.EVENT_LBUTTONUP: self.left_down=False
            if self.user and event == cv2.EVENT_RBUTTONUP: self.right_down = False
            if event == cv2.EVENT_MBUTTONDOWN: cv2.destroyAllWindows()

        WIN_NAME = 'draw_rect'
        cv2.namedWindow(WIN_NAME, 0)
        cv2.setMouseCallback(WIN_NAME, onmouse_draw_rect, self)
        cv2.imshow(WIN_NAME, temp_img)
        cv2.waitKey(0)

    def D(self, pi, det_sigma,inv_sigma, zn, u):
        t = np.mat([zn-u])
        return -np.log10(pi+self.eps)+0.5*np.log10(det_sigma+self.eps)+\
               (0.5*t*inv_sigma*t.T)[0,0]

    def show(self, data):
        sns.kdeplot(data[:, 0], shade=True, color="orange")
        plt.show()
        plt.close()

    def cal_function_D(self, data, alpha):
        r, l, _ = data.shape
        K = 5
        pi, u, sigma, det_sigma, inv_sigma = np.zeros((2, K)), np.zeros((2, K, 3)), np.zeros((2, K, 3, 3)) \
            , np.zeros((2, K)), np.zeros((2, K, 3, 3))
        ### 运用聚类的思想把数据聚成5类 ###
        cla_datas = [data[alpha==0], data[alpha==1]]
        # 把上面数据点分为两组（非监督学习）
        #clfs = [KMeans(n_clusters=K), KMeans(n_clusters=K)]
        clfs = [GaussianMixture(n_components=K), GaussianMixture(n_components=K)]
        labels = []
        for al in range(2):
            clfs[al].fit(cla_datas[al])
            labels.append(clfs[al].predict(cla_datas[al]))
        ### 开始计算每个类别的各个值！！！ ###
        for al in range(2):
            cla_data = cla_datas[al]
            for kk in range(K):
                cur_data = cla_data[labels[al] == kk]
                pi[al, kk] = cur_data.shape[0] / cla_data.shape[0]
                u[al, kk] = np.mean(cur_data, axis=0)
                sigma[al, kk] = np.cov(cur_data.T)
                det_sigma[al, kk] = np.linalg.det(sigma[al, kk])
                inv_sigma[al, kk] = np.linalg.inv(self.eps*np.random.rand(3,3)+sigma[al, kk])
        ### 之后对于每个点计算其属于两方的权重！！！ ###
        D_zero,D_one = np.zeros((r, l)), np.zeros((r,l))
        pbar = tqdm.tqdm(range(r))
        pbar.set_description('Caculating function D...')
        self.s_pbar.set('Caculating function D...')
        self.p1['value'] = 0
        final_pre_zero,final_pre_one = clfs[0].predict(data.reshape((r*l,3))).reshape((r,l)), \
                                       clfs[1].predict(data.reshape((r * l, 3))).reshape((r, l))
        for ii in pbar:
            self.p1['value'] += 100/r
            for kk in range(l):
                zero_l,one_l = final_pre_zero[ii,kk], final_pre_one[ii,kk]
                D_zero[ii,kk],D_one[ii,kk] = self.D(pi[0, zero_l], det_sigma[0, zero_l],inv_sigma[0, zero_l], data[ii,kk],
                                                    u[0, zero_l]),\
                                            self.D(pi[1, one_l], det_sigma[1, one_l], inv_sigma[1, one_l], data[ii, kk],
                                                   u[1, one_l])
        # ### 把那些确定是背景的图片权重置为0！！！ ###
        D_zero[alpha==0],D_one[alpha==0] = 0, 500
        D_zero[self.back],D_one[self.back] = 0,500 #用户标记的强行置为0
        D_zero[self.front],D_one[self.front]= 500,0 #用户标记的强行置为1 ???????
        return D_zero, D_one



    def cal_weights(self):
        gamma = 50
        temp_img = self.src_img.astype(np.float32)
        r, l, _ = temp_img.shape
        weights = np.zeros((r, l, 4))
        pbar = tqdm.tqdm(range(r))
        pbar.set_description('Caculating edge weight...')
        self.s_pbar.set('Caculating edge weight...')
        self.p1['value'] = 0
        for ii in pbar:
            self.p1['value'] += 100/r
            for kk in range(l):
                if kk > 0: #left
                    diff = temp_img[ii,kk]-temp_img[ii,kk-1]
                    weights[ii, kk, 0] = gamma * np.exp(-self.beta*diff.dot(diff))
                if ii > 0 and kk > 0: #up left
                    diff = temp_img[ii, kk] - temp_img[ii-1, kk - 1]
                    weights[ii, kk, 1] = gamma * np.exp(-self.beta*diff.dot(diff)) / math.sqrt(2)
                if ii > 0: # up
                    diff = temp_img[ii, kk] - temp_img[ii - 1, kk]
                    weights[ii, kk, 2] = gamma * np.exp(-self.beta*diff.dot(diff))
                if ii > 0 and kk < l - 1: #up right
                    diff = temp_img[ii, kk] - temp_img[ii - 1, kk+1]
                    weights[ii, kk, 3] = gamma * np.exp(-self.beta*diff.dot(diff)) / math.sqrt(2)
        return weights

    def cal_beta(self):
        temp_img = self.src_img.astype(np.float32)
        r, l, _ = temp_img.shape
        for ii in range(r):
            for kk in range(l):
                if kk > 0:
                    diff = temp_img[ii,kk]-temp_img[ii,kk-1]
                    self.beta += diff.dot(diff)
                if ii > 0 and kk > 0:
                    diff = temp_img[ii, kk] - temp_img[ii-1, kk - 1]
                    self.beta += diff.dot(diff)
                if ii > 0:
                    diff = temp_img[ii, kk] - temp_img[ii - 1, kk]
                    self.beta += diff.dot(diff)
                if ii > 0 and kk < l - 1:
                    diff = temp_img[ii, kk] - temp_img[ii - 1, kk+1]
                    self.beta += diff.dot(diff)
        self.beta = 1 / (2*self.beta/(4*r*l-3*r-3*l + 2))

    def construct_graph(self, D_zero, D_one, weights):
        temp_img = self.src_img.astype(np.float32)
        r, l, _ = temp_img.shape
        g = maxflow.Graph[float](r*l, 2*(4*r*l-3*r-3*l+2))
        nodeids = g.add_grid_nodes((r, l))
        g.add_grid_tedges(nodeids, D_zero, D_one)
        for ii in range(r):
            for kk in range(l):
                id = ii*l + kk
                assert id == nodeids[ii, kk]
                if kk > 0: g.add_edge(id, id-1, weights[ii,kk,0], weights[ii,kk,0])
                if ii > 0 and kk > 0: g.add_edge(id, id-l-1, weights[ii,kk,1], weights[ii,kk,1])
                if ii > 0: g.add_edge(id, id-l, weights[ii,kk,2], weights[ii,kk,2])
                if ii > 0 and kk < l - 1: g.add_edge(id, id-l+1, weights[ii,kk,3], weights[ii,kk,3])
        # Find the maximum flow.
        g.maxflow()
        # Get the segments of the nodes in the grid.
        return np.logical_not(g.get_grid_segments(nodeids))

    def matting(self):
        iteration = 1
        temp_img = self.src_img.astype(np.float32)
        r,l,_ = temp_img.shape
        ### 创建alpha矩阵  前景为1背景为0！！！！！！ ###
        alpha = np.zeros((r,l), dtype = np.bool)
        alpha[self.r1:self.r2, self.l1:self.l2] = 1 ### 初始化矩形区域 ###
        self.cal_beta()
        ### 计算两个边之间的距离！！！ ###
        weights = self.cal_weights()
        for it in range(iteration):
            ### 计算高斯混合分布的参数 ###
            D_zero, D_one = self.cal_function_D(temp_img, alpha)
            #self.check(D_zero, D_one)
            ### 开始最小流切割！！！ ###
            sgm = self.construct_graph(D_zero, D_one, weights)
            target_image = np.zeros((r,l,3))
            target_image[np.logical_not(sgm)] = [255,255,255]
            target_image[sgm] = self.src_img[sgm]
            cv2.imshow('Target Image', target_image.astype(np.uint8))
            cv2.waitKey(0)
            cv2.imwrite('resources/grabcut/res.png', target_image.astype(np.uint8))
            ### 更新alpha ###
            alpha = sgm

    def begin_matting(self):
        self.draw_rect()
        self.matting()

def move(source, target):
    target_image = cv2.imread(target)
    src_img = cv2.imread(source)
    vector = [-20, 200]
    count = 0
    for ii in range(src_img.shape[0]):
        for kk in range(src_img.shape[1]):
            if list(src_img[ii,kk]) != [255,255,255]:
                target_image[ii+vector[0], kk+vector[1]] = src_img[ii, kk]
    cv2.imwrite('target.png', target_image)

if __name__ == '__main__':
    ### gama 设定为了1 ###
    grabcut = GrabCut('resources/2.jpg')
    grabcut.begin_matting()

