import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_pro = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=12)
font_pro_min = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=10)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'
### K折交叉验证 ###
import tqdm

import cv2

def show_image(img):
    print(type(img))
    cv2.imshow('test', img)
    cv2.waitKey(0)

class PatchMatch:

    def __init__(self, path1, path2, s_pbar, p1):
        ### 读入两张图片 ###
        self.img1 = cv2.imread(path1)
        self.img2 = cv2.imread(path2)
        #resize = (400, 300)
        if self.img1.shape != self.img2.shape:
            ### rezise参数是宽x高！！！ ###
            self.img1 = cv2.resize(self.img1, (self.img2.shape[1], self.img2.shape[0]))
            # self.img2 = cv2.resize(self.img2, (self.img1.shape[1], self.img1.shape[0]))
            # self.img1 = cv2.resize(self.img1, resize)
            # self.img2 = cv2.resize(self.img2, resize)
        assert self.img1.shape == self.img2.shape
        ### 设定初始化参数：patch大小 (MxM) 一般都是3x3 或者 5x5 ###
        self.M = 5
        self.patch_rows, self.patch_cols = self.img1.shape[0]-self.M,self.img1.shape[1]-self.M
        ### 偏移矩阵 ###
        self.f = np.zeros((self.patch_rows, self.patch_cols, 2))
        ### 使用error来记录当前的差错 避免重复计算 ###
        self.error = np.zeros((self.patch_rows, self.patch_cols))
        ### 下面是Random Search操作 ###
        self.w = max([self.patch_rows, self.patch_cols])
        self.alpha = 0.5
        self.norm_type = cv2.NORM_L1

        self.s_pbar,self.p1 = s_pbar,p1

    ### 给f初始化 注意边界均初始化为0 所以边界均不动！！！！！ ###
    def initiate_f(self):
        pbar = tqdm.tqdm(range(1, self.patch_rows - 1))
        pbar.set_description('Initiate Offset Description f')
        self.s_pbar.set('Initiate Offset Description f')
        self.p1['value'] = 0
        for ii in pbar:
            self.p1['value'] += 100 / (self.patch_rows-2)
            for kk in range(1, self.patch_cols - 1):
                ### 每次给ii,kk附加一个当前的偏移offset！！！ ###
                i_off = np.random.randint(-ii, self.patch_rows - ii)
                k_off = np.random.randint(-kk, self.patch_cols - kk)
                self.f[ii, kk] = [i_off, k_off]
                x = ii + i_off
                y = kk + k_off
                self.error[ii, kk] = cv2.norm(self.img1[ii:ii+self.M, kk:kk+self.M]-
                        self.img2[x:x+self.M, y:y+self.M], self.norm_type)

    def cal_D(self,cur_x,cur_y, i1, k1, off):
        i2 = int(i1 + off[0])
        k2 = int(k1 + off[1])
        dis = cv2.norm(self.img1[cur_x:cur_x+self.M, cur_y:cur_y+self.M]-
                        self.img2[i2:i2+self.M, k2:k2+self.M], self.norm_type)
        ### 新的i2-cur_x k2-cur_y才是相对于当前的偏移！！！ ###
        return tuple(([i2-cur_x, k2-cur_y], dis))

    def propagate_and_random_search(self, k_iterator, which):
        rows, cols = self.patch_rows, self.patch_cols
        self.s_pbar.set('Propagate And Random Search on Iteration ' + str(k_iterator))
        self.p1['value'] = 0
        if which:
            ### 注意边界不能动！！！ ###
            pbar = tqdm.tqdm(range(1, rows - 1))
            pbar.set_description('Propagate And Random Search on Iteration ' + str(k_iterator))
            ### 从上至下 从左到右 ###
            for ii in pbar:
                self.p1['value'] += 100/(rows-2)
                for kk in range(1, cols - 1):
                    ### 比较f(ii,kk) f(ii-1,kk) f(ii,kk-1) ###
                    ### 错了错了 不是单纯比较f 而是要比较距离！！！！！ ###
                    ### 计算当前位置与其他的偏移啊！！！ ###
                    ans = sorted([(self.f[ii,kk], self.error[ii,kk]), self.cal_D(ii,kk,ii-1, kk, self.f[ii-1,kk]),
                                  self.cal_D(ii,kk,ii, kk-1, self.f[ii,kk-1])], key=lambda x:x[1])[0]
                    self.f[ii, kk] = ans[0]
                    self.error[ii, kk] = ans[1]
                    ### 立刻进行Random Search操作 ###
                    self.random_search(ii, kk)
        else:
            pbar = tqdm.tqdm(range(rows - 2, 0, -1))
            pbar.set_description('Propagate And Random Search on Iteration ' + str(k_iterator))
            ### 从下至上 从右到左 ###
            for ii in pbar:
                self.p1['value'] += 100 / (rows - 2)
                for kk in range(cols - 2, 0, -1):
                    ### 比较f(ii,kk) f(ii+1,kk) f(ii,kk+1) ###
                    ans = sorted([(self.f[ii,kk], self.error[ii,kk]), self.cal_D(ii,kk,ii + 1, kk, self.f[ii + 1, kk]),
                                  self.cal_D(ii,kk,ii, kk + 1, self.f[ii, kk + 1])], key=lambda x: x[1])[0]
                    self.f[ii, kk] = ans[0]
                    self.error[ii, kk] = ans[1]
                    ### 立刻进行Random Search操作 ###
                    self.random_search(ii, kk)

    def once_random_search(self, ii, kk):
        radius = self.w
        ### 应该是对匹配区域进行random search操作！！！ ###
        x_src = int(ii + self.f[ii, kk, 0])
        y_src = int(kk + self.f[ii, kk, 1])

        r_weight = np.random.uniform(-1, 1, size=2)
        ### 增加扰动 ###
        x = int(x_src + radius * r_weight[0])
        y = int(y_src + radius * r_weight[1])
        if x < 0 or x >= self.patch_rows or y < 0 or y >= self.patch_cols:
            return
        ### 计算offset距离 ###
        d_src = self.error[ii, kk]
        d = cv2.norm(self.img1[ii:ii + self.M, kk:kk + self.M] -
                     self.img2[x:x + self.M, y:y + self.M], self.norm_type)
        ### 更新距离 ###
        if d < d_src:
            self.f[ii, kk] = [x - ii, y - kk]
            self.error[ii, kk] = d

    def random_search(self, ii, kk):
        ### 应该是对匹配区域进行random search操作！！！ ###
        radius = self.w
        x_src = int(ii + self.f[ii, kk, 0])
        y_src = int(kk + self.f[ii, kk, 1])
        while radius > 1:
            r_weight = np.random.uniform(-1, 1, size=2)
            ### 增加扰动 ###
            x = int(x_src + radius * r_weight[0])
            y = int(y_src + radius * r_weight[1])
            if x < 0 or x >= self.patch_rows or y < 0 or y >= self.patch_cols:
                radius *= self.alpha
                continue
            ### 计算offset距离 ###
            d_src = self.error[ii, kk]
            d = cv2.norm(self.img1[ii:ii+self.M, kk:kk+self.M]-
                            self.img2[x:x+self.M, y:y+self.M], self.norm_type)
            ### 更新距离 ###
            if d < d_src:
                x_src,y_src = x,y
                self.f[ii, kk] = [x-ii, y-kk]
                self.error[ii, kk] = d
            ### 每次半径减半 ###
            radius *= self.alpha

    ### 开始match ###
    def begin_match(self):
        ### First Step: Initiate f ###
        self.initiate_f()
        ### Second Step: Iterate by P1,S1,P2,S2... ###
        k_iterator = 2
        for ii in range(k_iterator):
            ### 每进行一次迭代就绘制出图看一下！！！ ###
            self.visualize_result(ii)
            ### 首先进行Propagate操作 其次是Random Search操作 两者同时交互进行 ###
            self.propagate_and_random_search(ii + 1, ii % 2 == 0)
        self.visualize_result(k_iterator)

    def visualize_result(self, k_iterator):
        ### 声明同样大小图片 将偏移加过来！！！ ###
        test_img = np.zeros(self.img1.shape)
        for ii in range(self.patch_rows):
            for kk in range(self.patch_cols):
                x = int(ii + self.f[ii, kk, 0])
                y = int(kk + self.f[ii, kk, 1])
                test_img[ii:ii+self.M, kk:kk+self.M] = test_img[ii:ii+self.M, kk:kk+self.M]+\
                                                        self.img2[x:x+self.M,y:y+self.M]
        ### 一个点最多会被加 M x M 次！！！ ###
        test_img = test_img / (self.M * self.M)
        test_img = test_img.astype(np.uint8)
        cv2.imshow('Iteration:'+str(k_iterator), test_img)
        cv2.waitKey(0)
        cv2.imwrite('resources/patchmatch/res.png', test_img)

if __name__ == '__main__':
    #show_image('1.jpeg')

    ### 为啥这么慢？？？ ###
    patch_match = PatchMatch('1.jpeg', '2.jpeg')
    patch_match.begin_match()
