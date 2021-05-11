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


class SeamCarving:
    def __init__(self, path, s_pbar, p1):
        ### 读入图片 ###
        self.img1 = cv2.imread(path)
        self.r, self.l = self.img1.shape[:2]
        ### 找到target条线 然后进行复制！！ ###
        gray_image = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY)
        p1['value'] = 0
        self.p1 = p1
        s_pbar.set('Caculating Horizontal Seams...')
        self.Map_H = self.find_all_seam(gray_image, axis=0)
        s_pbar.set('Caculating Vertical Seams...')
        self.Map_V = self.find_all_seam(gray_image, axis=1)
        ### 这是移除图片所用的 ###
        self.x = []
        self.y = []

        ### 放大比例 ###
        self.rate = 0.3



    def find_all_seam(self, img, axis = 1):
        ### 使用M记录三个值即 (x,y)原图的x,y坐标 以及当前的和 ###
        ### 必须是灰度图！！！ ###
        assert len(img.shape) == 2
        ### 首先将其梯度都算出来，并且相加 ###
        energy = abs(cv2.Sobel(img, cv2.CV_64F, 1, 0)) + abs(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        ### axis = 1 代表找到一条竖着的线 ###
        r, l = img.shape
        ### 声明一个矩阵M专门记录坐标对应 ###
        M = np.zeros((r, l))
        M_loc = np.zeros((r, l, 2), dtype = np.int)
        Index_Map = np.zeros((r, l))
        ### 初始化M 的点对应坐标 ###
        for ii in range(r):
            for kk in range(l):
                M_loc[ii, kk] = [ii, kk]
        ### 更新优化旅程正式开始！！！ ###
        if axis == 1:
            ### M第一行就是原来的能量 ###
            M[0] = energy[0]
            ### 从第二行开始迭代 ###
            for ii in range(1, r):
                for jj in range(0, l):
                    if jj == 0:
                        min_up = min(M[ii - 1, jj], M[ii - 1, jj + 1])
                    elif jj == l - 1:
                        min_up = min(M[ii - 1, jj], M[ii - 1, jj - 1])
                    else:
                        min_up = min(M[ii - 1, jj], M[ii - 1, jj - 1], M[ii - 1, jj + 1])
                    M[ii, jj] = energy[ii, jj] + min_up
            def back_trace():
                if M.shape[1] == 1:
                    return list([0 for ii in range(r)])
                ### 倒着向上追踪 ###
                dic_optimal = [np.argmin(M[r - 1])]
                for ii in range(r - 2, -1, -1):
                    index = dic_optimal[-1]
                    ### 当前检查的是ii行 需要找的是当前index附近哪里最小！！！ ###
                    if index == 0:
                        dist = [(index, M[ii, index]), (index + 1, M[ii, index + 1])]
                    elif index == M.shape[1] - 1:
                        dist = [(index, M[ii, index]), (index - 1, M[ii, index - 1])]
                    else:
                        dist = [(index, M[ii, index]), (index - 1, M[ii, index - 1]), (index + 1, M[ii, index + 1])]
                    dic_optimal.append(sorted(dist, key=lambda x: x[1])[0][0])
                    #assert M[ii, dic_optimal[-1]]+energy[ii+1,index]==M[ii+1,index]
                s = 0
                for ii in range(r):
                    s += energy[r-1-ii, dic_optimal[ii]]
                #assert min(M[r-1]) == s
                return list(reversed(dic_optimal))
            def choose(i1,i2,i3, row):
                ans = []
                if i1 >= 0 and i1 < M.shape[1]: ans.append(M[row, i1])
                if i2 >= 0 and i2 < M.shape[1]: ans.append(M[row, i2])
                if i3 >= 0 and i3 < M.shape[1]: ans.append(M[row, i3])
                return sorted(ans)[0]
            ### 现在开始执行 ###
            pbar = tqdm.tqdm(range(1, M.shape[1]+1))
            pbar.set_description('Trying to find all seams...')
            for turn in pbar:
                self.p1['value'] += 50 / self.l
                opt = back_trace() # 里面存储了turn轮每一行需要删除点的坐标
                ### 记录点 ###
                for ii in range(r):
                    ### 寻找真实的点 ###
                    real_which = M_loc[ii, opt[ii]]
                    Index_Map[real_which[0], real_which[1]] = turn
                ### 更新从第二行开始的所有M ###
                for kk in range(1, r):
                    last_index = opt[kk - 1]
                    ### 每次更新M 下面一行将要更新的数字 是上面的index-2 index-1 index+1.....index-1 index+1 index+2 ###
                    update = [last_index-1,last_index,last_index+1]
                    update.remove(opt[kk])
                    ### 第一个要更新的是update[0] 第二个是update[1] ###
                    if update[0] >= 0 and update[0] < M.shape[1]:
                        M[kk, update[0]] = energy[kk, update[0]] + choose(last_index-2,last_index-1,last_index+1,kk-1)
                    if update[1] >= 0 and update[1] < M.shape[1]:
                        M[kk, update[1]] = energy[kk, update[1]] + choose(last_index - 1, last_index + 1,last_index + 2, kk - 1)
                ### 更新完了之后缩减M, M_loc, energy ###
                for ii in range(r):
                    M[ii, opt[ii]:-1] = M[ii, opt[ii]+1:]
                    M_loc[ii, opt[ii]:-1] = M_loc[ii, opt[ii] + 1:]
                    energy[ii, opt[ii]:-1] = energy[ii, opt[ii] + 1:]
                ### 删除对应的列 ###
                M = np.delete(M, -1, axis=1)
                M_loc = np.delete(M_loc, -1, axis=1)
                energy = np.delete(energy, -1, axis=1)
            self.show_seam_heat(Index_Map, axis)
        else:
            ### M第一列就是原来的能量 ###
            M[:, 0] = energy[:, 0]
            ### 从第二列开始迭代 ###
            for ii in range(1, l):
                for jj in range(0, r):
                    ### ii是每一列 jj是每一行 ###
                    if jj == 0:
                        min_left = min(M[jj, ii - 1], M[jj + 1, ii - 1])
                    elif jj == r - 1:
                        min_left = min(M[jj, ii - 1], M[jj - 1, ii - 1])
                    else:
                        min_left = min(M[jj, ii - 1], M[jj + 1, ii - 1], M[jj - 1, ii - 1])
                    M[jj, ii] = energy[jj, ii] + min_left
            def back_trace():
                if M.shape[0] == 1:
                    return list([0 for ii in range(l)])
                ### 倒着向左边追踪 ###
                dic_optimal = [np.argmin(M[:, l - 1])]
                for ii in range(l - 2, -1, -1):
                    index = dic_optimal[-1]
                    ### 当前检查的是ii列 需要找的是当前index附近哪里最小！！！ ###
                    if index == 0:
                        dist = [(index, M[index, ii]), (index + 1, M[index + 1, ii])]
                    elif index == M.shape[0] - 1:
                        dist = [(index, M[index, ii]), (index - 1, M[index - 1, ii])]
                    else:
                        dist = [(index, M[index, ii]), (index - 1, M[index - 1, ii]), (index + 1, M[index + 1, ii])]
                    dic_optimal.append(sorted(dist, key=lambda x: x[1])[0][0])
                return dic_optimal

            def choose(i1, i2, i3, line):
                ans = []
                if i1 >= 0 and i1 < M.shape[0]: ans.append(M[i1, line])
                if i2 >= 0 and i2 < M.shape[0]: ans.append(M[i2, line])
                if i3 >= 0 and i3 < M.shape[0]: ans.append(M[i3, line])
                return sorted(ans)[0]

            ### 现在开始执行 ###
            pbar = tqdm.tqdm(range(1, M.shape[0] + 1))
            pbar.set_description('Trying to find all horizontal seams...')
            for turn in pbar:
                self.p1['value'] += 50 / self.r
                opt = back_trace()  # 里面存储了turn轮每一行需要删除点的坐标
                ### 记录点 ###
                for ii in range(l):
                    ### 寻找真实的点 ###
                    real_which = M_loc[opt[ii], ii]
                    Index_Map[real_which[0], real_which[1]] = turn
                ### 更新从第二列开始的所有M ###
                for kk in range(1, l):
                    last_index = opt[kk - 1]
                    ### 每次更新M 下面一行将要更新的数字 是上面的index-2 index-1 index+1.....index-1 index+1 index+2 ###
                    update = [last_index - 1, last_index, last_index + 1]
                    update.remove(opt[kk])
                    ### 第一个要更新的是update[0] 第二个是update[1] ###
                    if update[0] >= 0 and update[0] < M.shape[0]:
                        M[update[0], kk] = energy[update[0], kk] + choose(last_index - 2, last_index - 1,
                                                                          last_index + 1, kk - 1)
                    if update[1] >= 0 and update[1] < M.shape[0]:
                        M[update[1], kk] = energy[update[1], kk] + choose(last_index - 1, last_index + 1,
                                                                          last_index + 2, kk - 1)
                ### 更新完了之后缩减M, M_loc, energy ###
                for ii in range(l):
                    M[opt[ii]:-1, ii] = M[opt[ii] + 1:, ii]
                    M_loc[opt[ii]:-1, ii] = M_loc[opt[ii] + 1:, ii]
                    energy[opt[ii]:-1, ii] = energy[opt[ii] + 1:, ii]
                ### 删除对应的列 ###
                M = np.delete(M, -1, axis=0)
                M_loc = np.delete(M_loc, -1, axis=0)
                energy = np.delete(energy, -1, axis=0)
            self.show_seam_heat(Index_Map, axis)

        return Index_Map

    def show_seam_heat(self, index_map, axis):
        plt.imshow(index_map, cmap=plt.cm.jet)
        plt.savefig('resources/seamcarving/heat'+str(axis)+'.png', dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close()

    def find_seam(self, img, axis = 1):
        ### 必须是灰度图！！！ ###
        assert len(img.shape) == 2
        ### 首先将其梯度都算出来，并且相加 ###
        energy = abs(cv2.Sobel(img, cv2.CV_64F, 1, 0)) + abs(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        ### 声明一个矩阵M专门记录和 ###
        M = np.zeros(img.shape)
        ### axis = 1 代表找到一条竖着的线 ###
        r,l = img.shape
        if axis == 1:
            ### M第一行就是原来的能量 ###
            M[0] = energy[0]
            ### 从第二行开始迭代 ###
            for ii in range(1, r):
                for jj in range(0, l):
                    if jj == 0:
                        min_up = min(M[ii-1,jj], M[ii-1,jj+1])
                    elif jj == l - 1:
                        min_up = min(M[ii - 1, jj], M[ii - 1, jj - 1])
                    else:
                        min_up = min(M[ii - 1, jj], M[ii - 1, jj - 1], M[ii-1,jj+1])
                    M[ii, jj] = energy[ii, jj] + min_up
            ### 倒着向上追踪 ###
            dic_optimal = {(r-1):np.argmin(M[r-1])}
            for ii in range(r-2,-1,-1):
                index = dic_optimal[ii+1]
                ### 当前检查的是ii行 需要找的是当前index附近哪里最小！！！ ###
                if index == 0:
                    dist = [(index, M[ii, index]), (index+1, M[ii, index+1])]
                elif index == l - 1:
                    dist = [(index, M[ii, index]), (index - 1, M[ii, index - 1])]
                else:
                    dist = [(index, M[ii, index]), (index - 1, M[ii, index - 1]), (index+1,M[ii,index+1])]
                dic_optimal[ii] = sorted(dist, key=lambda x: x[1])[0][0]
        else:
            ### M第一列就是原来的能量 ###
            M[:, 0] = energy[:, 0]
            ### 从第二列开始迭代 ###
            for ii in range(1, l):
                for jj in range(0, r):
                    ### ii是每一列 jj是每一行 ###
                    if jj == 0:
                        min_left = min(M[jj, ii-1], M[jj+1, ii-1])
                    elif jj == r - 1:
                        min_left = min(M[jj, ii-1], M[jj-1, ii-1])
                    else:
                        min_left = min(M[jj, ii-1], M[jj+1, ii-1], M[jj-1,ii - 1])
                    M[jj, ii] = energy[jj, ii] + min_left
            ### 倒着向左边追踪 ###
            dic_optimal = {(l - 1): np.argmin(M[:, l - 1])}
            for ii in range(l - 2, -1, -1):
                index = dic_optimal[ii + 1]
                ### 当前检查的是ii列 需要找的是当前index附近哪里最小！！！ ###
                if index == 0:
                    dist = [(index, M[index, ii]), (index + 1, M[index+1, ii])]
                elif index == l - 1:
                    dist = [(index, M[index, ii]), (index - 1, M[index-1, ii])]
                else:
                    dist = [(index, M[index, ii]), (index - 1, M[index-1, ii]), (index + 1, M[index+1, ii])]
                dic_optimal[ii] = sorted(dist, key=lambda x: x[1])[0][0]

        return dic_optimal

    def realtime_resizing(self, after, axis = 1):
        self.p1['value'] = 0
        if axis == 1:
            target_image = np.zeros((self.img1.shape[0], after, 3), dtype=np.uint8)
            turn = self.img1.shape[1] - after
            for ii in range(self.r):
                self.p1['value'] = (ii+1) / self.r
                target_image[ii] = self.img1[ii, self.Map_V[ii] > turn]
        else:
            target_image = np.zeros((after, self.img1.shape[1], 3), dtype=np.uint8)
            turn = self.img1.shape[0] - after
            for ii in range(self.l):
                self.p1['value'] = (ii + 1) / self.l
                target_image[:, ii] = self.img1[self.Map_H[:, ii] > turn, ii]
        return target_image

    def object_remove(self, type = 'common'):
        global target_image
        self.test()
        ### 计算水平方向的线图！！！ ###
        index_map = self.Map_V
        ### 计算区域里面有几个点 ###
        dic_remove = {}
        x1, x2 = self.x
        y1, y2 = self.y
        for ii in range(x1, x2):
            for kk in range(y1, y2):
                dic_remove[index_map[ii, kk]] = 1
        ### 对于所有出现的点开始移除！！！ ###
        remove_count = len(dic_remove.keys())
        pbar = tqdm.tqdm(range(self.r))
        pbar.set_description('Begin Removing...')
        if type == 'common':
            target_image = np.zeros((self.img1.shape[0], self.img1.shape[1] - remove_count, 3), dtype=np.uint8)
            for ii in pbar:
                indexes = []
                for kk in range(self.l):
                    if index_map[ii, kk] in dic_remove.keys(): indexes.append(False)
                    else: indexes.append(True)
                target_image[ii] = self.img1[ii, indexes]
        elif type == 'max':
            turn = max(dic_remove.keys())
            target_image = np.zeros((self.img1.shape[0], int(self.l - turn), 3), dtype=np.uint8)
            for ii in pbar:
                target_image[ii] = self.img1[ii, index_map[ii] > turn]
        cv2.imshow('After removed...', target_image)
        cv2.waitKey(0)
        cv2.imwrite('resources/seamcarving/remove_' + str(target_image.shape) + '.png', target_image)
        self.x.clear()
        self.y.clear()

    def seam_enlarge(self, target, axis = 1):
        if axis == 1:
            #assert target <= self.l * (1+self.rate)
            target_image = np.zeros((self.r, target, 3), dtype = np.float)
            ### 每一行 ###
            pbar = tqdm.tqdm(range(self.r))
            pbar.set_description('Begin Enlarging...')
            for ii in pbar:
                self.p1['value'] = (ii + 1) / self.r
                cur_index = 0
                for kk in range(self.l):
                    target_image[ii, cur_index] = self.img1[ii, kk]
                    if self.Map_V[ii, kk] <= target - self.l:
                        ### 需要复制 ###
                        cur_index += 1
                        if kk == self.l - 1:
                            target_image[ii, cur_index] = (self.img1[ii, kk].astype(np.float)+self.img1[ii, kk-1])/2
                        else:
                            target_image[ii, cur_index] = (self.img1[ii, kk].astype(np.float) + self.img1[ii, kk + 1]) / 2
                    cur_index += 1
        else:
            #assert target <= self.r * (1+self.rate)
            target_image = np.zeros((target, self.l, 3), dtype=np.float)
            ### 每一行 ###
            pbar = tqdm.tqdm(range(self.l))
            pbar.set_description('Begin Enlarging...')
            for ii in pbar:
                self.p1['value'] = (ii + 1) / self.l
                cur_index = 0
                for kk in range(self.r):
                    target_image[cur_index, ii] = self.img1[kk, ii]
                    if self.Map_H[kk, ii] <= target - self.r:
                        ### 需要复制 ###
                        cur_index += 1
                        if kk == self.r - 1:
                            target_image[cur_index, ii] = (self.img1[kk, ii].astype(np.float) + self.img1[
                                kk - 1, ii]) / 2
                        else:
                            target_image[cur_index, ii] = (self.img1[kk, ii].astype(np.float) + self.img1[
                                kk + 1, ii]) / 2
                    cur_index += 1
        target_image = target_image.astype(np.uint8)
        return target_image

    def update_image(self, now_image):
        print('begin update...')
        self.img1 = now_image
        self.r, self.l = self.img1.shape[:2]
        ### 找到target条线 然后进行复制！！ ###
        gray_image = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY)
        self.Map_H = self.find_all_seam(gray_image, axis=0)
        self.Map_V = self.find_all_seam(gray_image, axis=1)
        print('End Update...')

    def seam_multisize(self, target, s_pbar):
        def set_pbar(h, w):
            s_pbar.set('Resizing Image to ('+str(h)+','+str(w)+')')
        ### 两个维度 ###
        if target[0] == self.r and target[1] == self.l:
            assert 0
        which = 0
        while self.r != target[0] or self.l != target[1]:
            if target[0] == self.r:
                ### 只需要扩展列 ###
                set_pbar(target[0], target[1])
                if target[1] < self.l:target_image = self.realtime_resizing(after=target[1], axis=1)
                else: target_image = self.seam_enlarge(target=target[1], axis=1)
            elif target[1] == self.l:
                ### 只需要扩展列 ###
                set_pbar(target[0], target[1])
                if target[0] < self.r: target_image = self.realtime_resizing(after=target[0], axis=0)
                else: target_image = self.seam_enlarge(target=target[0], axis=0)
            else:
                ### 看目前该改变哪一个了 每次1.1倍进行改动 ###
                if which == 0:
                    ### 放大还是缩小 ###
                    if self.r < target[0]:
                        row_target = min(int((1+self.rate)*self.r), target[0])
                        ### 放大 ###
                        set_pbar(row_target, self.l)
                        target_image = self.seam_enlarge(row_target, axis=which)
                    else:
                        row_target = max(int((1-self.rate)* self.r), target[0])
                        ### 缩小 ###
                        set_pbar(row_target, self.l)
                        target_image = self.realtime_resizing(row_target, axis=which)
                else:
                    ### 放大还是缩小 ###
                    if self.l < target[1]:
                        line_target = min(int((1+self.rate)* self.l), target[1])
                        ### 放大 ###
                        set_pbar(self.r, line_target)
                        target_image = self.seam_enlarge(line_target, axis=which)
                    else:
                        line_target = max(int((1-self.rate)* self.l), target[1])
                        ### 缩小 ###
                        set_pbar(self.r, line_target)
                        target_image = self.realtime_resizing(line_target, axis=which)
            self.update_image(target_image)
            which = (which + 1) % 2
        cv2.imshow('After resized...', self.img1)
        cv2.waitKey(0)
        cv2.imwrite('resources/seamcarving/multi.png', self.img1)




    def seam_shorten_resize(self, after, axis = 1):
        ### 计算需要合并多少次呢 ###
        target_image = self.img1.copy()
        gray_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)
        def save_result(target_image):
            ### 查看结果 ###
            target_image = target_image.astype(np.uint8)
            cv2.imshow('After Resized...', target_image)
            cv2.waitKey(0)
            cv2.imwrite('resized_to_' + str(target_image.shape) + '.png', target_image)
        if axis == 1:
            shorten_count = self.img1.shape[1] - min(after)
            ### 开始合并 ###
            pbar = tqdm.tqdm(range(1, shorten_count+1))
            pbar.set_description('Begin Resize the Image...')
            for ii in pbar:
                ### 计算 Seam Line 第ii次削减了ii列 ###
                dic_optimal = self.find_seam(gray_image, axis)
                ### 将每一行删除对应的 像素点！！！ ###
                for (every_line,index) in dic_optimal.items():
                    gray_image[every_line, index:-1] = gray_image[every_line, index+1:]
                    target_image[every_line, index:-1] = target_image[every_line, index+1:]
                ### 赋值新的target ###
                target_image = np.delete(target_image, -1, axis=1)
                gray_image = np.delete(gray_image, -1, axis=1)
                if target_image.shape[1] in after:
                    save_result(target_image)
        else:
            ### 计算需要合并多少次呢 ###
            shorten_count = self.img1.shape[0] - min(after)
            ### 开始合并 ###
            pbar = tqdm.tqdm(range(shorten_count))
            pbar.set_description('Begin Resize the Image...')
            for ii in pbar:
                ### 计算 Seam Line ###
                dic_optimal = self.find_seam(gray_image, axis)
                ### 将每一列进行压缩删除对应的index 像素点！！！ ###
                for (every_col, index) in dic_optimal.items():
                    gray_image[index:-1, every_col] = gray_image[index+1:,every_col]
                    target_image[index:-1, every_col] = target_image[index + 1:, every_col]
                ### 赋值新的target ###
                target_image = np.delete(target_image, -1, axis=0)
                gray_image = np.delete(gray_image, -1, axis=0)
                if target_image.shape[0] in after:
                    save_result(target_image)

    def seam_muti_shorten(self, target_size):
        ### 需要使用一个数组T(r, c)记录变换到r c的最小值 ###
        ### T(r, c) T(r-1, c) 这时需要删除最佳行 或者 T(r, c-1)这时需要删除最佳列 ###
        r,c = self.r - target_size[0], self.l - target_size[1]
        T = np.zeros((r, c))
        ### 还需要记录一行图片 以及前一列的图片。。。 ###
        row_images = np.zeros((c-1, self.r, self.l), dtype = np.uint8)
        line_images = np.zeros((r-1, self.r, self.l), dtype = np.uint8)
        list_backtrace = []
        ### 首先计算列以及行 ###


        pass

    def common_resize(self, target, axis = 1):
        if axis == 1:
            target_image = cv2.resize(self.img1, (target, self.img1.shape[0]))
        else:
            target_image = cv2.resize(self.img1, (self.img1.shape[1], target))
        cv2.imshow('After Resized...', target_image)
        cv2.waitKey(0)
        cv2.imwrite('common_to_' + str(target_image.shape) + '.png', target_image)

    def test(self):
        ### 内联鼠标响应函数 ###
        def show_match(event, x, y, flags, param):
            # 鼠标回调函数定义
            if event == cv2.EVENT_LBUTTONDBLCLK:
                self.x.append(y)
                self.y.append(x)
                cv2.circle(circle_image, (x, y), radius=1, thickness=5, color=(0,0,255))
                cv2.imshow('image', circle_image)
                if len(self.x) == 2:
                    cv2.destroyAllWindows()
                ### 找出match图像 ###
                #cv2.circle(self.img1, (x, y), 50, (255, 0, 0), 1)
        circle_image = self.img1.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', show_match)
        cv2.imshow('image', self.img1)
        cv2.waitKey(0)


if __name__ == '__main__':
    seamcarving = SeamCarving(r'resources/11.jpg')



    ### 下面是来进行图片压缩的 ###
    #seamcarving.seam_shorten_resize(after=[200, 300], axis=0)
    #seamcarving.seam_shorten_resize(after=[500], axis=1)

    ### 下面是大量优化之后的情况 ###
    #seamcarving.realtime_resizing(after = 200, axis = 0)
    ### 图像放大 ###
    #seamcarving.seam_enlarge(target=370, axis=0)
    seamcarving.seam_multisize(target=[250, 700])


    ### 物体移除 ###
    seamcarving.object_remove(type='common')

    ### opencv resized 平凡算法 ###
    #seamcarving.common_resize(target=300, axis=1)
    #seamcarving.common_resize(target=200, axis=1)
    #seamcarving.common_resize(target=200, axis=0)
    #seamcarving.common_resize(target=300, axis=0)

