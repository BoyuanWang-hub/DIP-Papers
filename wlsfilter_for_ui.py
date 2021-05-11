import numpy as np
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
### 稀疏矩阵 ###
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

class WLSFilter:
    def __init__(self, path, s_pbar, p1):
        self.img = cv2.imread(path)
        self.src_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)
        self.img1 = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.img1 = self.img1.astype(np.float32)
        self.r, self.l = self.img1.shape

        self.eps = 1e-5
        self.path = path

        self.s_pbar,self.p1 = s_pbar,p1

    def show_local_extrema(self, row, middle, save_path = None):
        plt.plot(range(self.l), self.src_img[row], c='pink')
        plt.plot(range(self.l), middle[row], c='black')
        if save_path != None:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()

    def begin_filter(self, alpha, lanbuta, l):
        ### 此算法的重点是解一个线性方程组 (I + lanbuta * Lg)u = g ###
        ### 其中Lg = Dx.T * Ax * Dx + Dy.T * Ay *Dy ###
        ### Dx.T * Ax * Dx 对角线上基础元素是aii,kk  如果ii处于1-倒数第二行之间 则再加上aii-1,kk ###
        ###### 如果cur_id - n存在 则前面是 -aii-1,kk 如果后面存在 则后面是 -aii,kk #####
        ### 其中 Dx Dy是向前差分矩阵 ###
        ### 现在开始构造Ax=b的形式 ###
        ### 三元组来存 稀疏矩阵 ###
        ### 求出x,y方向上的a ###
        log_img = np.log10(self.img1 + self.eps)
        Ax = np.abs(np.append(np.diff(log_img,1,0), np.zeros((1, self.l)), axis=0))
        Ax[-1] = Ax[-2]
        Ay = np.abs(np.append(np.diff(log_img,1,1), np.zeros((self.r, 1)), axis=1))
        Ay[:, -1] = Ay[:, -2]
        Ax = Ax**alpha + self.eps
        Ay = Ay**alpha + self.eps
        Ax = lanbuta * ( Ax**(-1) )
        Ay = lanbuta * ( Ay**(-1) )
        row = []
        col = []
        data = []
        b = np.zeros((self.r * self.l, 1))
        pbar = tqdm.tqdm(range(self.r))
        pbar.set_description('Caculating Matrix A and b...')
        self.s_pbar.set(l)
        self.p1['value'] = 0
        for ii in pbar:
            self.p1['value'] += 100/self.r
            for kk in range(self.l):
                cur_id = ii * self.l + kk
                b[cur_id] = self.img1[ii, kk]
                ### 首先来一个自己 ###
                center = 1 + Ax[ii,kk] + Ay[ii,kk]
                ### 其次处理Dx.T * Ax * Dx ###
                if ii > 0 and ii < self.r - 1: center += Ax[ii-1, kk]
                if ii > 0:
                    row.append(cur_id)  ### 前向偏移 ###
                    col.append(cur_id - self.l)
                    data.append(-Ax[ii - 1, kk])
                if ii < self.r - 1:
                    row.append(cur_id)  ### 后向偏移 ###
                    col.append(cur_id + self.l)
                    data.append(-Ax[ii, kk])
                ### 下面处理后面的 Dy.T * Ay * Dy ###
                if kk > 0 and kk < self.l - 1: center += Ay[ii, kk-1]
                if kk > 0:
                    row.append(cur_id)  ### 前向偏移 ###
                    col.append(cur_id - 1)
                    data.append(-Ay[ii, kk-1])
                if kk < self.l - 1:
                    row.append(cur_id)  ### 后向偏移 ###
                    col.append(cur_id + 1)
                    data.append(-Ay[ii, kk])
                ### 最后补上中心 ###
                row.append(cur_id)
                col.append(cur_id)
                data.append(center)
        ### 求解方程组并返回 ###
        self.s_pbar.set('Solving Equations...')
        sp_A = csc_matrix((data, (row, col)), shape=(self.r * self.l, self.r * self.l))
        return spsolve(sp_A, b).reshape((self.r, self.l))

    def begin_wls(self, alpha, lanbuta):
        target_image = self.begin_filter(alpha, lanbuta).astype(np.uint8)
        self.show_local_extrema(row=check, middle=target_image)
        d1 = self.img1 - target_image
        cv2.imshow('Detail 1', d1)
        cv2.waitKey(0)
        cv2.imshow('target', target_image)
        cv2.waitKey(0)
        cv2.imwrite('resources/alpha='+str(alpha)+'lambda='+str(lanbuta)+'.png', target_image)

    def begin_slowly(self ,alpha, start, end):
        count = 10
        every = (end - start) / count
        for ii in range(count):
            cur_lanbuta = start + ii * every
            print(cur_lanbuta)
            target_image = self.begin_filter(alpha, cur_lanbuta).astype(np.uint8)
            cv2.imwrite('resources/Iteration' + str(ii+1) + '.png', target_image)

    def prin(self):
        s = ''
        for ii in range(10, 0, -1):
            s += '"Iteration'+str(ii)+'.png" '
        s += '"' + self.path[self.path.index('/')+1:] + '"'
        print(s)

    def three_canels_wls(self, alpha, lanbuta):
        all_channels = cv2.imread(self.path)
        target_image = np.zeros(all_channels.shape)
        for ii in range(3):
            self.img1 = all_channels[:, :, ii].astype(np.float32)
            target_image[:, :, ii] = self.begin_filter(alpha, lanbuta)
        target_image = target_image.astype(np.uint8)
        cv2.imshow('3 Channels', target_image)
        cv2.waitKey(0)
        cv2.imwrite('3_channels1.png', target_image)

    def ui_boost(self):
        ### 0 for coarse   1 for medium  2 for fine ###
        all_channels = cv2.imread(self.path)
        lab_image = cv2.cvtColor(all_channels, cv2.COLOR_BGR2LAB)
        b_coarse = lab_image.copy().astype(np.float32)
        d_fine = lab_image.copy().astype(np.float32)
        self.img1 = lab_image[:, :, 0]
        b_coarse[:, :, 0] = self.begin_filter(alpha=1.4, lanbuta=0.4, l = 'Caculating Coarse...')
        d_fine[:, :, 0] = self.begin_filter(alpha=1.2, lanbuta=0.1, l = 'Caculating Fine...')
        d_medium = d_fine - b_coarse
        d_fine = lab_image - d_fine
        self.b_coarse,self.d_medium,self.d_fine = b_coarse,d_medium,d_fine
        cv2.imwrite('resources/wls/res.png', cv2.cvtColor(b_coarse.astype(np.uint8), cv2.COLOR_LAB2BGR))
        cv2.imshow('result', cv2.imread('resources/wls/res.png'))
        cv2.waitKey(0)

    def enhance(self):
        WIN_NAME = 'ENHANCE'
        cv2.namedWindow(WIN_NAME)
        self.yita,self.duota0,self.duota1,self.duota2 = 1,1,1,1
        def show():
            self.g_bar = self.tone_manipulation(lab_image,
                                           [self.d_fine[..., 0], self.d_medium[..., 0], self.b_coarse[..., 0]],
                                           [self.duota2, self.duota1, self.duota0], self.yita)
            self.g_bar = cv2.cvtColor(self.g_bar, cv2.COLOR_LAB2BGR)
            cv2.imshow(WIN_NAME, self.g_bar)
        def yita_change(x):
            self.yita = x / 50
            show()
        def duota0_change(x):
            self.duota0 = x
            show()
        def duota1_change(x):
            self.duota1 = x
            show()
        def duota2_change(x):
            self.duota2 = x
            show()
        cv2.createTrackbar('Control Light',WIN_NAME,50,100,yita_change)
        cv2.createTrackbar('Duota0 For Coarse', WIN_NAME, 1, 20, duota0_change)
        cv2.createTrackbar('Duota1 For Medium', WIN_NAME, 1, 20, duota1_change)
        cv2.createTrackbar('Duota2 For Fine', WIN_NAME, 1, 20, duota2_change)
        all_channels = cv2.imread(self.path)
        lab_image = cv2.cvtColor(all_channels, cv2.COLOR_BGR2LAB)
        cv2.imshow(WIN_NAME, all_channels)
        cv2.waitKey(0)
        cv2.imwrite('resources/wls/boost.png', self.g_bar)

    # def detail_boost(self, duota0, duota1, duota2):
    #     ### 0 for coarse   1 for medium  2 for fine ###
    #     all_channels = cv2.imread(self.path)
    #     lab_image = cv2.cvtColor(all_channels, cv2.COLOR_BGR2LAB)
    #     b_coarse = np.zeros(lab_image.shape)
    #     d_fine = np.zeros(lab_image.shape)
    #     for ii in range(3):
    #         self.img1 = lab_image[:,:,ii]
    #         b_coarse[:,:,ii] = self.begin_filter(alpha=1.4, lanbuta=0.4)
    #         d_fine[:,:,ii] = self.begin_filter(alpha=1.2, lanbuta=0.1)
    #     d_medium = d_fine - b_coarse
    #     d_fine = lab_image - d_fine
    #     while True:
    #         s = input().split(' ')
    #         yita,duota0,duota1,duota2 = float(s[0]),float(s[1]),float(s[2]),float(s[3])
    #         g_bar,fine = self.tone_manipulation(lab_image,[d_fine[...,0],d_medium[...,0], b_coarse[...,0]],[duota2, duota1, duota0],yita)
    #         g_bar = cv2.cvtColor(g_bar, cv2.COLOR_LAB2BGR)
    #         fine = cv2.cvtColor(fine, cv2.COLOR_LAB2BGR)
    #         cv2.imshow('final', g_bar)
    #         cv2.imshow('fine', fine)
    #         cv2.waitKey(0)
    #         cv2.imwrite('Boost.png', g_bar)


    def sigmoid(self, a, im):
        im = im.astype(np.float32)
        return 1 / (1 + np.exp(-a * im)) - 0.5

    def tone_manipulation(self, lab, sequence, a_values, exposure=1.0, L_min=0, L_max=255):
        lab = lab.astype(np.float32)
        L_mean = np.mean(lab)
        normalized_sequence = []
        ### 将其归一化到 -0.5 --- 0.5 ###
        for s, L in enumerate(sequence):
            L = L.copy()
            if s == (len(sequence) - 1): ### 与此同时 将最后一个直接减去L_mean ###
                L = L*exposure - L_mean
            L_normalized = (L - L_min) / (L_max - L_min)
            normalized_sequence.append(L_normalized)
        ### 将其处理成sigmoid ###
        nonlinear_sequence = []
        for i, (L_normalized, a) in enumerate(zip(normalized_sequence, a_values)):
            nonlinearity = self.sigmoid(a, L_normalized)
            nonlinear_sequence.append(nonlinearity)
        ### 将其转回来 !!! ###
        denormalized_sequence = []
        for n, nonlinear_L in enumerate(nonlinear_sequence):
            nonlinear_L = nonlinear_L * (L_max - L_min) + L_min
            if n == (len(sequence) - 1):
                nonlinear_L += L_mean
            denormalized_sequence.append(nonlinear_L)
        ### 最后将其叠加起来 ###
        L_new = np.zeros_like(lab, dtype=np.float32)
        for d, denormalized_L in enumerate(denormalized_sequence):
            L_new[..., 0] += denormalized_L
        L_new[..., 1:] = lab[..., 1:]
        L_new = np.clip(L_new, a_min=L_min, a_max=L_max)  # 限制在0-255之间
        return L_new.astype(np.uint8)

if __name__ == '__main__':
    check = 100

    wlsfilter = WLSFilter('resources/6.jpg')

    #wlsfilter.begin_wls(alpha = 1.8, lanbuta = 0.35)

    #wlsfilter.three_canels_wls(alpha=1.8, lanbuta=2.5)
    #wlsfilter.three_canels_wls(alpha=1.8, lanbuta=10)

    #wlsfilter.begin_slowly(alpha = 1.2, start = 0.1, end = 1.3)

    #wlsfilter.prin()

    wlsfilter.detail_boost(duota0=1, duota1=1, duota2=1)

    # a = np.array([[1,2,3],[5,5,5]])
    # b = a.copy()
    # b[0,0] = 99
    # print(a,b)
    #a = ['1.2', '3.3', '5.5']
    #cv2.normalize(a,a,alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
    #print(a.dtype)