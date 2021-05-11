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
import matplotlib.animation as animation

class Decomposition:
    def __init__(self, path, constant = False, s_pbar = None, p1 = None):
        self.img = cv2.imread(path)
        self.img1 = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.img1 = self.img1.astype(np.float32)
        self.r, self.l = self.img1.shape
        ### 初始值设定为3 ###
        self.k = 3
        ### sigma矩阵 ###
        self.sigma = np.zeros((self.r, self.l))
        self.src_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)
        self.path = path[:path.index('.')]
        self.abs_path = path
        self.constant = constant
        self.s_pbar,self.p1 = s_pbar, p1

    def get_all_extrema(self, max_e = True):
        ### k x k 区域的“极值”点 ###
        half = int (self.k / 2)
        list_extrema = {}
        pbar = tqdm.tqdm(range(self.r))
        pbar.set_description('Caculating Extrema...')
        for ii in pbar:
            for kk in range(self.l):
                l = max(0, kk - half)
                r = min(self.l-1, kk+half)
                u = max(0, ii - half)
                d = min(self.r - 1, ii + half)
                ### l-r u-d区域内 ###
                if max_e:
                    ### 统计比我大的点 ###
                    count = np.count_nonzero(self.img1[u:d+1, l:r + 1] > self.img1[ii,kk])
                else:
                    ### 统计比我小的点 ###
                    count = np.count_nonzero(self.img1[u:d+1, l:r + 1] < self.img1[ii, kk])
                ### 比我大的点不超过k-1个我就是极值 比我小的网络不超过k-1 ###
                if count <= self.k - 1:
                    list_extrema[(ii, kk)] = 1
        pbar.close()
        return list_extrema

    def show_extrema(self, list_extrema):
        target_image = np.zeros((self.r, self.l), dtype = np.uint8)
        for (x,y) in list_extrema.keys():
            target_image[x, y] = self.img1[x, y]
        cv2.imshow('extrema', target_image)
        cv2.waitKey(0)
        plt.imshow(target_image, cmap=plt.cm.jet)
        plt.show()
        plt.close()

    ### 图片强度 ###
    def show_intensity(self, row):
        plt.plot(range(self.l), self.img1[row], c = 'red')
        plt.show()
        plt.close()

    ### 图片强度 + 局部极值 ###
    def show_intensity_with_extrema(self, row, max_extrema, min_extrema):
        plt.plot(range(self.l), self.img1[row], c='pink')
        for ii in range(self.l):
            if (row, ii) in max_extrema.keys():
                plt.scatter(ii, self.img1[row, ii], c = 'blue', s = 5)
            if (row, ii) in min_extrema.keys():
                plt.scatter(ii, self.img1[row, ii], c='red', s=5)
        plt.show()
        plt.close()

    ### 绘制计算极值之后的 极大值 极小值 以及均值图片 ###
    def show_local_extrema(self, row, middle, save_path):
        plt.plot(range(self.l), self.src_img[row], c='pink')
        # plt.plot(range(self.l), E_max[row], c='blue')
        # plt.plot(range(self.l), E_min[row], c='purple')
        plt.plot(range(self.l), middle[row], c='black')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()

    ### 计算sigma 局部方差 ###
    def cal_sigma(self):
        half = int(self.k / 2)
        pbar = tqdm.tqdm(range(self.r))
        pbar.set_description('Caculating Part Sigma...')
        self.s_pbar.set('Caculating Part Sigma...')
        self.p1['value'] = 0
        for ii in pbar:
            self.p1['value'] += 100 / self.r
            for kk in range(self.l):
                if not self.constant:
                    l = max(0, kk - half)
                    r = min(self.l - 1, kk + half)
                    u = max(0, ii - half)
                    d = min(self.r - 1, ii + half)
                    ### l-r u-d区域内 ###
                    cur_var = np.std(self.img1[u:d+1, l:r + 1])
                    if cur_var == 0:
                        self.sigma[ii, kk] = 0
                    else: self.sigma[ii, kk] = -1 / (2 * cur_var)
                else: self.sigma[ii, kk] = -0.01
        pbar.close()

    ### 利用插值法来计算 E ###
    def cal_E(self, extrema):
        ### 三元组来存 稀疏矩阵 ###
        row = []
        col = []
        data = []
        ### 此题目就是求解线性方程组 Ax = b ###
        ### 每个E都是其所有邻居的加权平均 E(r) = Wrs * E(s) 建立一个(mn, k)矩阵即可 ###
        half = int(self.k / 2)
        b = np.zeros((self.r*self.l, 1))
        pbar = tqdm.tqdm(range(self.r))
        pbar.set_description('Caculating Matrix A and b...')
        self.s_pbar.set('Caculating Matrix A and b...')
        self.p1['value'] = 0
        for ii in pbar:
            self.p1['value'] += 100/self.r
            for kk in range(self.l):
                cur_id = ii * self.l + kk
                ### 直接给当前点赋值A 1 ###
                row.append(cur_id)
                col.append(cur_id)
                data.append(1)
                if (ii, kk) in extrema.keys():
                    ### 直接给b赋值 ###
                    b[cur_id, 0] = self.img1[ii, kk]
                else:
                    ### 计算相邻的点 ###
                    l = max(0, kk - half)
                    r = min(self.l - 1, kk + half)
                    u = max(0, ii - half)
                    d = min(self.r - 1, ii + half)
                    s = 1e-5
                    W = []
                    for i in range(u, d+1):
                        for k in range(l, r+1):
                            if i == ii and k == kk: continue
                            W.append(np.exp((int(self.img1[ii, kk]) - int(self.img1[i, k])) ** 2 * self.sigma[ii, kk]))
                            s += W[-1]
                    index = 0
                    for i in range(u, d+1):
                        for k in range(l, r+1):
                            if i == ii and k == kk: continue
                            ### 计算当前权重 ###
                            w_ik = W[index] / s
                            index += 1
                            if (i,k) in extrema.keys():
                                ### 常量 ###
                                b[cur_id, 0] += w_ik * self.img1[i, k]
                            else:
                                ### 给矩阵 A 赋值 ###
                                row.append(cur_id)
                                col.append(i*self.l + k)
                                data.append(-w_ik)
        pbar.close()
        ### 求解方程组并返回 ###
        self.s_pbar.set('Solving Equations...')
        sp_A = csc_matrix((data, (row, col)), shape=(self.r*self.l, self.r*self.l))
        return spsolve(sp_A, b).reshape((self.r, self.l))

    def begin_decomposition(self, iteration, add, demo = True):
        res = []
        for ii in range(iteration):
            max_extrema = self.get_all_extrema(max_e=True)
            min_extrema = self.get_all_extrema(max_e=False)
            if demo: self.show_intensity_with_extrema(check, max_extrema, min_extrema)
            ### 计算局部方差 ###
            self.cal_sigma()
            ### 计算两者 ###
            E_max_extrema = self.cal_E(max_extrema)
            E_min_extrema = self.cal_E(min_extrema)
            ### 显示出来 ###
            target_image = (E_max_extrema + E_min_extrema) / 2
            if demo: self.show_local_extrema(check, target_image, self.path+'Smooth'+str(ii+1)+'.png')
            target_image = target_image.astype(np.uint8)
            res.append(target_image)
            if demo:
                cv2.imshow('target image', target_image)
                cv2.waitKey(0)
                cv2.imwrite(self.path+'Iteration'+str(ii+1)+'.png', target_image)
            ### 更新 ###
            self.img1 = target_image
            self.k += add
        return res

    def show_result(self, iteration):
        goal_r, goal_l = self.r*2, self.l*2
        fig,ax = plt.subplots()
        ims = []
        for ii in range(iteration+1):
            if ii == 0:
                img = cv2.imread('3.png')
                img = cv2.resize(img, (goal_r, goal_l))
                t = ax.text(0.5, 1.05, "Raw Image",
                                size=plt.rcParams["axes.titlesize"],
                                ha="center", transform=ax.transAxes, fontproperties = font_pro)
            else:
                img = cv2.imread('Iteration'+str(ii)+'.png')
                img = cv2.resize(img, (goal_r, goal_l))
                t = ax.text(0.5, 1.05, "After {} Iteration(s)".format(ii),
                                size=plt.rcParams["axes.titlesize"],
                                ha="center", transform=ax.transAxes, fontproperties = font_pro)
            #plt.tick_params(axis=u'both', which=u'both', length=0)
            ax.axis('off')
            im = ax.imshow(img)
            ims.append([im, t])
        ani = animation.ArtistAnimation(fig, ims, interval=500)
        ani.save("test.gif")

        fig, ax = plt.subplots()
        ims = []
        for ii in range(iteration):
            img = cv2.imread('Smooth' + str(ii+1) + '.png')
            img = cv2.resize(img, (goal_r, goal_l))
            t = ax.text(0.5, 1.05, "After {} Iteration(s)".format(ii+1),
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, fontproperties=font_pro)
            ax.axis('off')
            im = ax.imshow(img)
            ims.append([im, t])
        ani = animation.ArtistAnimation(fig, ims, interval=300, repeat_delay=1000)
        ani.save("test1.gif", writer='pillow')

    def show_Dn(self, n = 1):
        cur_D_sum = np.zeros((self.r, self.l))
        for ii in range(1, n+1):
            Mii = cv2.cvtColor(cv2.imread(self.path+'Iteration'+str(ii)+'.png'), cv2.COLOR_RGB2GRAY)
            next_sum = self.src_img - Mii
            Dii = next_sum - cur_D_sum
            cur_D_sum = next_sum
        cv2.imshow('D'+str(n), Dii.astype(np.uint8))
        cv2.waitKey(0)

    def ui_boost(self):
        all_channels = cv2.imread(self.abs_path)
        lab_image = cv2.cvtColor(all_channels, cv2.COLOR_BGR2LAB)
        self.img1 = lab_image[:, :, 0]
        res = self.begin_decomposition(2, 2, False)
        self.b_coarse = lab_image.copy().astype(np.float32)
        self.d_fine = lab_image.copy().astype(np.float32)
        self.b_coarse[:,:,0] = res[1].astype(np.float32)
        self.d_fine[:,:,0] = res[0].astype(np.float32)
        self.d_medium = self.d_fine - self.b_coarse
        self.d_fine = lab_image - self.d_fine
        cv2.imwrite('resources/decomposition/res.png', cv2.cvtColor(self.b_coarse.astype(np.uint8), cv2.COLOR_LAB2BGR))
        cv2.imshow('result', cv2.imread('resources/decomposition/res.png'))
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
        all_channels = cv2.imread(self.abs_path)
        lab_image = cv2.cvtColor(all_channels, cv2.COLOR_BGR2LAB)
        cv2.imshow(WIN_NAME, all_channels)
        cv2.waitKey(0)
        cv2.imwrite('resources/decomposition/boost.png', self.g_bar)

    def begin_tone(self, iteration, add):
        res = self.begin_decomposition(iteration,add,False)
        all_channels = cv2.imread(self.abs_path)
        lab_image = cv2.cvtColor(all_channels, cv2.COLOR_BGR2LAB)
        b_coarse = res[-1].astype(np.float32)
        d_fine = (self.src_img-res[0]).astype(np.float32)
        d_medium = (res[0]-res[1]).astype(np.float32)
        d_final = (res[1] - res[2]).astype(np.float32)
        while True:
            print('Please Input:')
            s = input().split(' ')
            assert len(s) == 2
            yita, duota = float(s[0]), float(s[1])
            g_bar, fine = self.tone_manipulation(lab_image, [d_fine, d_medium, d_final, b_coarse],
                                                 [duota, duota, duota, duota], yita)
            g_bar = cv2.cvtColor(g_bar, cv2.COLOR_LAB2BGR)
            fine = cv2.cvtColor(fine, cv2.COLOR_LAB2BGR)
            cv2.imshow('final', g_bar)
            cv2.imshow('fine', fine)
            cv2.waitKey(0)
            cv2.imwrite('Boost.png', g_bar)


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

    ### 固定参数下 -0.01比较好 变动参数下：标准差比较好 ###
    decomposition = Decomposition('66.jpg', constant = True)
    #decomposition.begin_decomposition(iteration = 3, add = 2)
    decomposition.begin_tone(3,2)
    #decomposition.show_result(9)
    #decomposition.show_Dn(n= 3)