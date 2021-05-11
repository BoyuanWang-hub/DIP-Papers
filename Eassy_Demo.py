import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_pro = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=12)
font_pro_min = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=10)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'
import cv2
import tkinter
from tkinter import filedialog
import tkinter.messagebox
import os
from SeamCarving_for_ui import SeamCarving
from patchmatch_for_ui import PatchMatch
from imagewarp_for_ui import ImageWarping
from wlsfilter_for_ui import WLSFilter
from Decomposition_for_ui import Decomposition
from GrabCut_for_ui import GrabCut
from tkinter import ttk
from tkinter import HORIZONTAL
from threading import Timer
from PIL import Image,ImageTk


def get_img_obj(file, width):
    photo = Image.open(file)
    alpha = width / photo.size[0]
    r, l = photo.size[0], photo.size[1]
    photo = photo.resize((int(photo.size[0] * alpha), int(photo.size[1] * alpha)))
    return photo, r, l

### 第一篇 ###
class PatchMatch_UI:
    def __init__(self):
        self.first_width = 280
        self.src_img = False
        self.tar_img = False
        self.src_path, self.tar_path = '', ''
        self.res = False

    def get_image(self, src, s_image, is_src):
        img_path = str(filedialog.askopenfilename(title=u'选择文件', initialdir='resources',
                                              filetypes=[('Images', '*.png *.jpg *.jpeg')]))
        if img_path == '':
            tkinter.messagebox.showinfo('Warning！', 'You have not input any images!!!')
            return
        index = img_path.index('resources')
        photo,l,r = get_img_obj(img_path[index:], self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        src.config(image=photo)
        src.image = photo  # keep a reference
        s_image.set('(' + str(r)+','+str(l)+')')
        if is_src:
            self.src_path = img_path[index:]
            self.src_img = True
        else:
            self.tar_path = img_path[index:]
            self.tar_img = True

    def thread_start(self, *args):
        dic_obj = args[0]
        s_pbar = dic_obj['s_pbar']
        p1 = dic_obj['p1']
        patchmatch_obj = PatchMatch(self.src_path, self.tar_path, s_pbar, p1)
        patchmatch_obj.begin_match()
        del patchmatch_obj
        self.res = True


    def begin(self, s_pbar, p1):
        if not self.src_img or not self.tar_img:
            tkinter.messagebox.showinfo('警告！', '必须先输入源图和目标图！')
            return
        t = Timer(0, self.thread_start, [{'s_pbar':s_pbar, 'p1':p1}])
        t.start()

    def show(self):
        if not self.res:
            tkinter.messagebox.showinfo('警告！', '结果图还没出！')
            return
        cv2.imshow('result', cv2.imread('resources/patchmatch/res.png'))
        cv2.waitKey(0)

    def on_exit(self):
        self.src_w.destroy()
        del self
        window_main()

    def how_to_use(self):
        s = 'First Step: 点击Choose Src Image选择源图；\n'+ \
            'Second Step: 点击Choose Tar Image选择匹配图；\n' + \
            'Third Step: 点击Begin PatchMatch开始匹配。\n' + \
            'Final: 匹配结果可以点击Show Result查看！！！'
        tkinter.messagebox.showinfo('使用说明', s)

    def begin_patchmatch(self, src_w):
        src_w.destroy()
        demo_window = tkinter.Tk()
        self.src_w = demo_window
        demo_window.protocol("WM_DELETE_WINDOW", self.on_exit)
        # ###重写×控件####
        screenwidth = demo_window.winfo_screenwidth()
        screenheight = demo_window.winfo_screenheight()
        demo_window.geometry(
            '%dx%d+%d+%d' % (700, 400, (screenwidth - 500 - 150) / 2, (screenheight - 400 - 150) / 2))
        demo_window.title('Patch Match')

        ### Source Imaeg 原图存放 ###
        s_image = tkinter.StringVar(demo_window)
        s_image.set('Source Image')
        src_label = tkinter.Label(textvariable=s_image, fg='black', font=('Times', 15))
        src_label.place(x=140, y=150)
        photo, _, _ = get_img_obj('resources/11.jpg', self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        img_label = tkinter.Label(demo_window, imag=photo)
        img_label.place(x=50, y=200)

        ### Target Imaeg 目标图存放 ###
        t_image = tkinter.StringVar(demo_window)
        t_image.set('Target Image')
        tar_label = tkinter.Label(textvariable=t_image, fg='black', font=('Times', 15))
        tar_label.place(x=450, y=150)
        t_label = tkinter.Label(demo_window, imag=photo)
        t_label.place(x=350, y=200)

        choose_src = tkinter.Button(demo_window, text='Choose Src Image', fg='black', font=('Times', 12))
        choose_src.bind('<Button-1>', lambda event: self.get_image(img_label, s_image, True))
        choose_src.place(x=130, y=110)

        choose_tar = tkinter.Button(demo_window, text='Choose Tar Image', fg='black', font=('Times', 12))
        choose_tar.bind('<Button-1>', lambda event: self.get_image(t_label, t_image, False))
        choose_tar.place(x=440, y=110)

        begin = tkinter.Button(demo_window, text='Begin PatchMatch', fg='black', font=('Times', 12))
        begin.bind('<Button-1>', lambda event: self.begin(s_pbar, p1))
        begin.place(x=290, y=100)

        show = tkinter.Button(demo_window, text='Show Result', fg='black', font=('Times', 12))
        show.bind('<Button-1>', lambda event: self.show())
        show.place(x=300, y=140)

        ### 进度条 ###
        p1 = ttk.Progressbar(demo_window, length=600, mode='determinate', orient=HORIZONTAL)
        p1.place(x=50, y=65)
        p1['maximum'] = 100
        p1['value'] = 0
        ### 显示进度条的信息 ###
        s_pbar = tkinter.StringVar(demo_window)
        s_pbar.set('Progress Will Be Shown Here...')
        pbar_label = tkinter.Label(demo_window, textvariable=s_pbar, font=('Times', 12))
        pbar_label.place(x=250, y=20)

        ##########主菜单是menu##################
        menu = tkinter.Menu(demo_window)
        demo_window.config(menu=menu)
        first_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='PatchMatch使用说明', menu=first_menu)
        first_menu.add_command(label='PatchMatch使用说明', command=self.how_to_use)
        second_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='开发者', menu=second_menu)
        second_menu.add_command(label='关于开发者', command=developer)

        demo_window.mainloop()

### 第二篇 ###
class SeamCarve_UI:
    def __init__(self):
        self.is_image = False
        self.first_width = 350
        self.second_width = 250
        self.seam_obj = None
        self.path = ''
        self.multi_path = 'resources/seamcarving/multi.png'
        self.busy = False

    def thread_start_seam(self, *args):
        dic_obj = args[0]
        path = dic_obj['path']
        s_pbar = dic_obj['s_pbar']
        hon = dic_obj['hon']
        ver = dic_obj['ver']
        ### 初始化 ###
        self.busy = True
        self.seam_obj = SeamCarving(path, s_pbar, dic_obj['p1'])
        self.busy = False
        ### 放置图片！！！ ###
        photo,_,_ = self.get_img_obj('resources/seamcarving/heat0.png', self.second_width)
        photo = ImageTk.PhotoImage(image=photo)
        hon.config(image=photo)
        hon.image = photo  # keep a reference

        photo1,_,_ = self.get_img_obj('resources/seamcarving/heat1.png', self.second_width)
        photo1 = ImageTk.PhotoImage(image=photo1)
        ver.config(image=photo1)
        ver.image = photo1  # keep a reference

    def thread_start_resize(self, *args):
        dic_obj = args[0]
        target = dic_obj['target']
        s_pbar = dic_obj['s_pbar']
        self.busy = True
        self.seam_obj.seam_multisize(target=target, s_pbar=s_pbar)
        self.busy = False

    def get_image(self, src, hon, ver, s_pbar, p1, s_image):
        img_path = str(filedialog.askopenfilename(title=u'选择文件', initialdir='resources',
                                              filetypes=[('Images', '*.png *.jpg *.jpeg')]))
        if img_path == '':
            tkinter.messagebox.showinfo('Warning！', 'You have not input any images!!!')
            return
        index = img_path.index('resources')
        self.path = img_path[index:]
        photo,l,r = self.get_img_obj(img_path[index:], self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        src.config(image=photo)
        src.image = photo  # keep a reference
        s_image.set('Src Image (' + str(r)+','+str(l)+')')
        t = Timer(0, self.thread_start_seam, [{'path':img_path[index:], 's_pbar':s_pbar, 'p1':p1,
                                               'hon':hon, 'ver':ver}])
        t.start()
        self.is_image = True

    def resize_image(self, input_H, input_W, s_pbar, p1):
        if not self.is_image or self.busy:
            tkinter.messagebox.showinfo('Warning！', 'Can not Execute this Command!!!')
            return
        try:
            next_H = int(input_H.get())
            next_W = int(input_W.get())
        except:
            tkinter.messagebox.showinfo('Warning！', 'You must input H and W as Integers!!!')
            return
        t = Timer(0, self.thread_start_resize, [{'target':[next_H, next_W], 's_pbar':s_pbar}])
        t.start()



    def remove_obj(self):
        if not self.is_image or self.busy:
            tkinter.messagebox.showinfo('Warning！', 'Can not Execute this Command!!!')
            return
        self.seam_obj.object_remove()

    def get_img_obj(self, file, width):
        photo = Image.open(file)
        alpha = width / photo.size[0]
        r,l = photo.size[0], photo.size[1]
        photo = photo.resize((int(photo.size[0]*alpha), int(photo.size[1]*alpha)))
        return photo,r,l

    def view_result(self):
        if not os.path.isfile(self.multi_path):
            tkinter.messagebox.showinfo('Warning！', 'Result Gone!!!')
            return
        cv2.imshow('Result', cv2.imread(self.multi_path))
        cv2.waitKey(0)

    def view_src(self):
        if not os.path.isfile(self.path):
            tkinter.messagebox.showinfo('Warning！', 'Source Image not choosed!!!')
            return
        cv2.imshow('Source Image', cv2.imread(self.path))
        cv2.waitKey(0)

    def on_exit(self):
        self.src_w.destroy()
        del self
        window_main()

    def how_to_use(self):
        s = 'First Step: 点击Choose Image选择目标图片，之后会自动计算该图片水平和竖直方向的Seam；\n'+ \
            'Second Step: 输入重定向图片的高H与W，之后点击Resize；\n' + \
            'Third Step: 点击View Result查看结果。\n'
        tkinter.messagebox.showinfo('使用说明', s)

    def begin_seamcarve(self, src_w):
        src_w.destroy()
        carve_window = tkinter.Tk()
        self.src_w = carve_window
        carve_window.protocol("WM_DELETE_WINDOW", self.on_exit)
        # ###重写×控件####
        screenwidth = carve_window.winfo_screenwidth()
        screenheight = carve_window.winfo_screenheight()
        carve_window.geometry(
            '%dx%d+%d+%d' % (900, 600, (screenwidth - 900 - 150) / 2, (screenheight - 600 - 150) / 2))
        carve_window.title('Seam Carve for Image Retargeting')

        ### Source Imaeg 原图存放 ###
        s_image = tkinter.StringVar(carve_window)
        s_image.set('Source Image')
        src_label = tkinter.Label(textvariable=s_image, fg='black', font=('Times', 15))
        src_label.place(x=150, y=250)
        photo,_,_ = self.get_img_obj('resources/11.jpg', self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        img_label = tkinter.Label(carve_window, imag=photo)
        img_label.place(x=50, y=300)

        ### Source Imaeg 水平图存放 ###
        hon_label = tkinter.Label(text='Honrizontal Seams', fg='black', font=('Times', 15))
        hon_label.place(x=440, y=250)
        photo1,_,_ = self.get_img_obj('resources/11.jpg', self.second_width)
        photo1 = ImageTk.PhotoImage(image=photo1)
        hon_img = tkinter.Label(carve_window, imag=photo1)
        hon_img.place(x=410, y=300)

        ### Source Imaeg 竖直图存放 ###
        ver_label = tkinter.Label(text='Vertical Seams', fg='black', font=('Times', 15))
        ver_label.place(x=660, y=250)
        ver_img = tkinter.Label(carve_window, imag=photo1)
        ver_img.place(x=610, y=300)

        btn1 = tkinter.Button(carve_window, text='Choose Image', fg='black', font=('Times', 15))
        btn1.bind('<Button-1>', lambda event: self.get_image(img_label, hon_img, ver_img, s_pbar, p1,
                                                             s_image))
        btn1.place(x=30, y=60)
        ### 进度条 ###
        p1 = ttk.Progressbar(carve_window, length=600, mode='determinate', orient=HORIZONTAL)
        p1.place(x=200, y=65)
        p1['maximum'] = 100
        p1['value'] = 0
        ### 显示进度条的信息 ###
        s_pbar = tkinter.StringVar(carve_window)
        s_pbar.set('Progress Will Be Shown Here...')
        pbar_label = tkinter.Label(carve_window, textvariable=s_pbar, font=('Times', 15))
        pbar_label.place(x=400, y=20)

        btn2 = tkinter.Button(carve_window, text='Resize Image to (H,W)', fg='black', font=('Times', 15))
        btn2.bind('<Button-1>', lambda event: self.resize_image(input_H, input_W, s_pbar, p1))
        btn2.place(x=50, y=120)
        input_H = tkinter.Entry(carve_window, font=('Times', 12), width=5)
        input_H.place(x=90, y=165, height=50)
        input_W = tkinter.Entry(carve_window, font=('Times', 12), width=5)
        input_W.place(x=150, y=165, height=50)

        btn_result = tkinter.Button(carve_window, text='View Result', fg='black', font=('Times', 16))
        btn_result.bind('<Button-1>', lambda event: self.view_result())
        btn_result.place(x=300, y=120)

        btn_src = tkinter.Button(carve_window, text='View Raw Image', fg='black', font=('Times', 16))
        btn_src.bind('<Button-1>', lambda event: self.view_src())
        btn_src.place(x=300, y=200)

        btn3 = tkinter.Button(carve_window, text='Remove Object', fg='black', font=('Times', 15))
        btn3.bind('<Button-1>', lambda event: self.remove_obj())
        btn3.place(x=500, y=120)

        ##########主菜单是menu##################
        menu = tkinter.Menu(carve_window)
        carve_window.config(menu=menu)
        first_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='SeamCarving使用说明', menu=first_menu)
        first_menu.add_command(label='SeamCarving使用说明', command=self.how_to_use)
        second_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='开发者', menu=second_menu)
        second_menu.add_command(label='关于开发者', command=developer)

        carve_window.mainloop()

### 第三篇 ###
class ImageWarp_UI:
    def __init__(self):
        self.first_width = 280
        self.src_img = False
        self.src_path = ''
        self.width = 500
        self.res = False

    def get_image(self, src):
        img_path = str(filedialog.askopenfilename(title=u'选择文件', initialdir='resources',
                                                  filetypes=[('Images', '*.png *.jpg *.jpeg')]))
        if img_path == '':
            tkinter.messagebox.showinfo('Warning！', 'You have not input any images!!!')
            return
        index = img_path.index('resources')
        photo, l, r = get_img_obj(img_path[index:], self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        src.config(image=photo)
        src.image = photo  # keep a reference
        self.src_path = img_path[index:]
        self.src_img = True

    def thread_start(self, *args):
        dic_obj = args[0]
        s_pbar = dic_obj['s_pbar']
        p1 = dic_obj['p1']
        tar = dic_obj['tar']
        warp_obj = ImageWarping(self.src_path, self.width, s_pbar, p1)
        warp_obj.begin_warp(False)
        self.res = True
        del warp_obj
        photo, l, r = get_img_obj('resources/Warping/res.png', self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        tar.config(image=photo)
        tar.image = photo  # keep a reference




    def begin(self, s_pbar, p1, tar):
        if not self.src_img:
            tkinter.messagebox.showinfo('警告！', '必须先输入源图！')
            return
        t = Timer(0, self.thread_start, [{'s_pbar': s_pbar, 'p1': p1, 'tar':tar}])
        t.start()

    def show(self):
        if not self.res:
            tkinter.messagebox.showinfo('警告', '结果图还没出！！！')
            return
        cv2.imshow('result', cv2.imread('resources/Warping/res.png'))
        cv2.waitKey(0)

    def on_exit(self):
        self.src_w.destroy()
        del self
        window_main()

    def how_to_use(self):
        s = 'First Step: 点击Choose Src Image选择源图；\n'+ \
            'Second Step: 点击Begin Warping开始扭曲，扭曲方法如下：\n\n单击鼠标左键选择起始点，' \
            '之后单机鼠标左键选择目标点(这样构成一对儿点)，可以选择多对儿点，选择完成后，双击鼠标右键开始算法！；\n\n' + \
            'Final: 匹配结果可以点击Show Result查看！！！'
        tkinter.messagebox.showinfo('使用说明', s)

    def begin_warp(self, src_w):
        src_w.destroy()
        demo_window = tkinter.Tk()
        self.src_w = demo_window
        demo_window.protocol("WM_DELETE_WINDOW", self.on_exit)
        # ###重写×控件####
        screenwidth = demo_window.winfo_screenwidth()
        screenheight = demo_window.winfo_screenheight()
        demo_window.geometry(
            '%dx%d+%d+%d' % (700, 400, (screenwidth - 500 - 150) / 2, (screenheight - 400 - 150) / 2))
        demo_window.title('Image Warping')

        ### Source Imaeg 原图存放 ###
        s_image = tkinter.StringVar(demo_window)
        s_image.set('Source Image')
        src_label = tkinter.Label(textvariable=s_image, fg='black', font=('Times', 15))
        src_label.place(x=140, y=150)
        photo, _, _ = get_img_obj('resources/11.jpg', self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        img_label = tkinter.Label(demo_window, imag=photo)
        img_label.place(x=50, y=200)

        ### Target Imaeg 目标图存放 ###
        t_image = tkinter.StringVar(demo_window)
        t_image.set('Target Image')
        tar_label = tkinter.Label(textvariable=t_image, fg='black', font=('Times', 15))
        tar_label.place(x=450, y=150)
        t_label = tkinter.Label(demo_window, imag=photo)
        t_label.place(x=350, y=200)

        choose_src = tkinter.Button(demo_window, text='Choose Src Image', fg='black', font=('Times', 12))
        choose_src.bind('<Button-1>', lambda event: self.get_image(img_label))
        choose_src.place(x=130, y=110)

        begin = tkinter.Button(demo_window, text='Begin Warping', fg='black', font=('Times', 12))
        begin.bind('<Button-1>', lambda event: self.begin(s_pbar, p1, t_label))
        begin.place(x=290, y=100)

        show = tkinter.Button(demo_window, text='Show Result', fg='black', font=('Times', 12))
        show.bind('<Button-1>', lambda event: self.show())
        show.place(x=300, y=140)

        ### 进度条 ###
        p1 = ttk.Progressbar(demo_window, length=600, mode='determinate', orient=HORIZONTAL)
        p1.place(x=50, y=65)
        p1['maximum'] = 100
        p1['value'] = 0
        ### 显示进度条的信息 ###
        s_pbar = tkinter.StringVar(demo_window)
        s_pbar.set('Progress Will Be Shown Here...')
        pbar_label = tkinter.Label(demo_window, textvariable=s_pbar, font=('Times', 12))
        pbar_label.place(x=250, y=20)

        ##########主菜单是menu##################
        menu = tkinter.Menu(demo_window)
        demo_window.config(menu=menu)
        first_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='Image Warping使用说明', menu=first_menu)
        first_menu.add_command(label='Image Warping使用说明', command=self.how_to_use)
        second_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='开发者', menu=second_menu)
        second_menu.add_command(label='关于开发者', command=developer)

        demo_window.mainloop()

### 第四篇 ###
class WLSFilter_UI:
    def __init__(self):
        self.first_width = 280
        self.src_path = ''
        self.wls_obj = None
        self.src_img = False
        self.can_enhance = False
        self.res1, self.res2 = False, False

    def get_image(self, src, s_image):
        img_path = str(filedialog.askopenfilename(title=u'选择文件', initialdir='resources',
                                              filetypes=[('Images', '*.png *.jpg *.jpeg')]))
        if img_path == '':
            tkinter.messagebox.showinfo('Warning！', 'You have not input any images!!!')
            return
        index = img_path.index('resources')
        photo,l,r = get_img_obj(img_path[index:], self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        src.config(image=photo)
        src.image = photo  # keep a reference
        s_image.set('(' + str(r)+','+str(l)+')')
        self.src_path = img_path[index:]
        self.src_img = True

    def thread_start(self, *args):
        dic_obj = args[0]
        s_pbar = dic_obj['s_pbar']
        p1 = dic_obj['p1']
        tar = dic_obj['tar']
        self.wls_obj = WLSFilter(self.src_path, s_pbar, p1)
        self.wls_obj.ui_boost()
        self.can_enhance = True
        photo, l, r = get_img_obj('resources/wls/res.png', self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        tar.config(image=photo)
        tar.image = photo  # keep a reference
        self.res1 = True


    def begin(self, s_pbar, p1, tar):
        if not self.src_img:
            tkinter.messagebox.showinfo('警告！', '必须先输入源图和目标图！')
            return
        t = Timer(0, self.thread_start, [{'s_pbar':s_pbar, 'p1':p1, 'tar':tar}])
        t.start()

    def show(self):
        if not self.res1:
            tkinter.messagebox.showinfo('警告！！！', '还没出图！！！')
            return
        cv2.imshow('result', cv2.imread('resources/wls/res.png'))
        cv2.waitKey(0)

    def show_boost(self):
        if not self.res2:
            tkinter.messagebox.showinfo('警告！！！', '还没出图！！！')
            return
        cv2.imshow('result', cv2.imread('resources/wls/boost.png'))
        cv2.waitKey(0)

    def on_exit(self):
        self.src_w.destroy()
        del self
        window_main()

    def enhance(self):
        if not self.can_enhance:
            tkinter.messagebox.showinfo('警告！！！', '请先执行WLS操作！！！')
            return
        self.wls_obj.enhance()
        self.res2 = True

    def how_to_use(self):
        s = 'First Step: 点击Choose Src Image选择源图；\n'+ \
            'Second Step: 点击Begin WLS执行模糊算法；\n' + \
            'Third Step: 点击Detail Enhance开始细节增强；\n' + \
            'Final: 匹配结果可以点击Show Result查看！！！'
        tkinter.messagebox.showinfo('使用说明', s)

    def begin_wls(self, src_w):
        src_w.destroy()
        demo_window = tkinter.Tk()
        self.src_w = demo_window
        demo_window.protocol("WM_DELETE_WINDOW", self.on_exit)
        # ###重写×控件####
        screenwidth = demo_window.winfo_screenwidth()
        screenheight = demo_window.winfo_screenheight()
        demo_window.geometry(
            '%dx%d+%d+%d' % (700, 400, (screenwidth - 500 - 150) / 2, (screenheight - 400 - 150) / 2))
        demo_window.title('WLS Filter')

        ### Source Imaeg 原图存放 ###
        s_image = tkinter.StringVar(demo_window)
        s_image.set('Source Image')
        src_label = tkinter.Label(textvariable=s_image, fg='black', font=('Times', 15))
        src_label.place(x=140, y=150)
        photo, _, _ = get_img_obj('resources/11.jpg', self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        img_label = tkinter.Label(demo_window, imag=photo)
        img_label.place(x=50, y=200)

        ### Target Imaeg 目标图存放 ###
        t_image = tkinter.StringVar(demo_window)
        t_image.set('Target Image')
        tar_label = tkinter.Label(textvariable=t_image, fg='black', font=('Times', 15))
        tar_label.place(x=450, y=150)
        t_label = tkinter.Label(demo_window, imag=photo)
        t_label.place(x=350, y=200)

        choose_src = tkinter.Button(demo_window, text='Choose Src Image', fg='black', font=('Times', 12))
        choose_src.bind('<Button-1>', lambda event: self.get_image(img_label, s_image))
        choose_src.place(x=130, y=110)

        choose_tar = tkinter.Button(demo_window, text='Show Result', fg='black', font=('Times', 12))
        choose_tar.bind('<Button-1>', lambda event: self.show())
        choose_tar.place(x=440, y=110)

        begin = tkinter.Button(demo_window, text='Begin WLS', fg='black', font=('Times', 12))
        begin.bind('<Button-1>', lambda event: self.begin(s_pbar, p1, t_label))
        begin.place(x=300, y=100)

        choose_tar1 = tkinter.Button(demo_window, text='Show Boosted', fg='black', font=('Times', 12))
        choose_tar1.bind('<Button-1>', lambda event: self.show_boost())
        choose_tar1.place(x=540, y=110)

        show = tkinter.Button(demo_window, text='Detail Enhance', fg='black', font=('Times', 12))
        show.bind('<Button-1>', lambda event: self.enhance())
        show.place(x=300, y=140)

        ### 进度条 ###
        p1 = ttk.Progressbar(demo_window, length=600, mode='determinate', orient=HORIZONTAL)
        p1.place(x=50, y=65)
        p1['maximum'] = 100
        p1['value'] = 0
        ### 显示进度条的信息 ###
        s_pbar = tkinter.StringVar(demo_window)
        s_pbar.set('Progress Will Be Shown Here...')
        pbar_label = tkinter.Label(demo_window, textvariable=s_pbar, font=('Times', 12))
        pbar_label.place(x=250, y=20)

        ##########主菜单是menu##################
        menu = tkinter.Menu(demo_window)
        demo_window.config(menu=menu)
        first_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='WLS使用说明', menu=first_menu)
        first_menu.add_command(label='WLS使用说明', command=self.how_to_use)
        second_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='开发者', menu=second_menu)
        second_menu.add_command(label='关于开发者', command=developer)

        demo_window.mainloop()

### 第五篇 ###
class Decomposition_UI:
    def __init__(self):
        self.first_width = 280
        self.src_path = ''
        self.decomp_obj = None
        self.src_img = False
        self.can_enhance = False
        self.res1, self.res2 = False, False

    def get_image(self, src, s_image):
        img_path = str(filedialog.askopenfilename(title=u'选择文件', initialdir='resources',
                                                  filetypes=[('Images', '*.png *.jpg *.jpeg')]))
        if img_path == '':
            tkinter.messagebox.showinfo('Warning！', 'You have not input any images!!!')
            return
        index = img_path.index('resources')
        photo, l, r = get_img_obj(img_path[index:], self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        src.config(image=photo)
        src.image = photo  # keep a reference
        s_image.set('(' + str(r) + ',' + str(l) + ')')
        self.src_path = img_path[index:]
        self.src_img = True

    def thread_start(self, *args):
        dic_obj = args[0]
        s_pbar = dic_obj['s_pbar']
        p1 = dic_obj['p1']
        tar = dic_obj['tar']
        self.decomp_obj = Decomposition(self.src_path, True, s_pbar, p1)
        self.decomp_obj.ui_boost()
        photo, l, r = get_img_obj('resources/decomposition/res.png', self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        tar.config(image=photo)
        tar.image = photo  # keep a reference
        self.res1 = True
        self.can_enhance = True

    def begin(self, s_pbar, p1, tar):
        if not self.src_img:
            tkinter.messagebox.showinfo('警告！', '必须先输入源图和目标图！')
            return
        t = Timer(0, self.thread_start, [{'s_pbar': s_pbar, 'p1': p1, 'tar': tar}])
        t.start()

    def show(self):
        if not self.res1:
            tkinter.messagebox.showinfo('警告！！！', '还没出图！！！')
            return
        cv2.imshow('result', cv2.imread('resources/decomposition/res.png'))
        cv2.waitKey(0)

    def show_boost(self):
        if not self.res2:
            tkinter.messagebox.showinfo('警告！！！', '还没出图！！！')
            return
        cv2.imshow('boost', cv2.imread('resources/decomposition/boost.png'))
        cv2.waitKey(0)

    def on_exit(self):
        self.src_w.destroy()
        del self
        window_main()

    def enhance(self):
        if not self.can_enhance:
            tkinter.messagebox.showinfo('警告！！！', '必须先执行EP Filter！！！')
            return
        self.decomp_obj.enhance()
        self.res2 = True

    def how_to_use(self):
        s = 'First Step: 点击Choose Src Image选择源图；\n'+ \
            'Second Step: 点击Begin EP Filter执行模糊算法；\n' + \
            'Third Step: 点击Detail Enhance开始细节增强；\n' + \
            'Final: 匹配结果可以点击Show Result查看！！！'
        tkinter.messagebox.showinfo('使用说明', s)

    def begin_decomposition(self, src_w):
        src_w.destroy()
        demo_window = tkinter.Tk()
        self.src_w = demo_window
        demo_window.protocol("WM_DELETE_WINDOW", self.on_exit)
        # ###重写×控件####
        screenwidth = demo_window.winfo_screenwidth()
        screenheight = demo_window.winfo_screenheight()
        demo_window.geometry(
            '%dx%d+%d+%d' % (700, 400, (screenwidth - 500 - 150) / 2, (screenheight - 400 - 150) / 2))
        demo_window.title('Edge-Preserving Filter')

        ### Source Imaeg 原图存放 ###
        s_image = tkinter.StringVar(demo_window)
        s_image.set('Source Image')
        src_label = tkinter.Label(textvariable=s_image, fg='black', font=('Times', 15))
        src_label.place(x=140, y=150)
        photo, _, _ = get_img_obj('resources/11.jpg', self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        img_label = tkinter.Label(demo_window, imag=photo)
        img_label.place(x=50, y=200)

        ### Target Imaeg 目标图存放 ###
        t_image = tkinter.StringVar(demo_window)
        t_image.set('Target Image')
        tar_label = tkinter.Label(textvariable=t_image, fg='black', font=('Times', 15))
        tar_label.place(x=450, y=150)
        t_label = tkinter.Label(demo_window, imag=photo)
        t_label.place(x=350, y=200)

        choose_src = tkinter.Button(demo_window, text='Choose Src Image', fg='black', font=('Times', 12))
        choose_src.bind('<Button-1>', lambda event: self.get_image(img_label, s_image))
        choose_src.place(x=130, y=110)

        choose_tar = tkinter.Button(demo_window, text='Show Result', fg='black', font=('Times', 12))
        choose_tar.bind('<Button-1>', lambda event: self.show())
        choose_tar.place(x=440, y=110)

        choose_tar1 = tkinter.Button(demo_window, text='Show Boosted', fg='black', font=('Times', 12))
        choose_tar1.bind('<Button-1>', lambda event: self.show_boost())
        choose_tar1.place(x=540, y=110)

        begin = tkinter.Button(demo_window, text='Begin EP Filter', fg='black', font=('Times', 12))
        begin.bind('<Button-1>', lambda event: self.begin(s_pbar, p1, t_label))
        begin.place(x=300, y=100)

        show = tkinter.Button(demo_window, text='Detail Enhance', fg='black', font=('Times', 12))
        show.bind('<Button-1>', lambda event: self.enhance())
        show.place(x=300, y=140)

        ### 进度条 ###
        p1 = ttk.Progressbar(demo_window, length=600, mode='determinate', orient=HORIZONTAL)
        p1.place(x=50, y=65)
        p1['maximum'] = 100
        p1['value'] = 0
        ### 显示进度条的信息 ###
        s_pbar = tkinter.StringVar(demo_window)
        s_pbar.set('Progress Will Be Shown Here...')
        pbar_label = tkinter.Label(demo_window, textvariable=s_pbar, font=('Times', 12))
        pbar_label.place(x=250, y=20)

        ##########主菜单是menu##################
        menu = tkinter.Menu(demo_window)
        demo_window.config(menu=menu)
        first_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='Edge Preserving Filter使用说明', menu=first_menu)
        first_menu.add_command(label='Edge Preserving Filter使用说明', command=self.how_to_use)
        second_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='开发者', menu=second_menu)
        second_menu.add_command(label='关于开发者', command=developer)

        demo_window.mainloop()

### 第六篇 ###
class GrabCut_UI:
    def __init__(self):
        self.first_width = 280
        self.src_path = ''
        self.grab_obj = None
        self.src_img = False
        self.res = False

    def get_image(self, src, s_image):
        img_path = str(filedialog.askopenfilename(title=u'选择文件', initialdir='resources',
                                                  filetypes=[('Images', '*.png *.jpg *.jpeg')]))
        if img_path == '':
            tkinter.messagebox.showinfo('Warning！', 'You have not input any images!!!')
            return
        index = img_path.index('resources')
        photo, l, r = get_img_obj(img_path[index:], self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        src.config(image=photo)
        src.image = photo  # keep a reference
        s_image.set('(' + str(r) + ',' + str(l) + ')')
        self.src_path = img_path[index:]
        self.src_img = True

    def thread_start(self, *args):
        dic_obj = args[0]
        s_pbar = dic_obj['s_pbar']
        p1 = dic_obj['p1']
        tar = dic_obj['tar']
        self.grabcut_obj = GrabCut(self.src_path, s_pbar, p1)
        self.grabcut_obj.begin_matting()
        photo, l, r = get_img_obj('resources/grabcut/res.png', self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        tar.config(image=photo)
        tar.image = photo  # keep a reference
        self.res = True

    def begin(self, s_pbar, p1, tar):
        if not self.src_img:
            tkinter.messagebox.showinfo('警告！', '必须先输入源图和目标图！')
            return
        t = Timer(0, self.thread_start, [{'s_pbar': s_pbar, 'p1': p1, 'tar': tar}])
        t.start()

    def show(self):
        if not self.res:
            tkinter.messagebox.showinfo('警告！！！', '还没出图！！！')
            return
        cv2.imshow('result', cv2.imread('resources/grabcut/res.png'))
        cv2.waitKey(0)

    def on_exit(self):
        self.src_w.destroy()
        del self
        window_main()

    def how_to_use(self):
        s = 'First Step: 点击Choose Src Image选择源图；\n'+ \
            'Second Step: 点击Begin GrabCut开始裁剪，裁剪时：\n\n1、首先拖动鼠标左键画出裁剪的矩形区域\n' \
            '2、之后拖动鼠标左键画出你认为的一定属于前景的部分(红色线，可以画多条)\n' \
            '3、之后拖动鼠标右键画出你认为的一定属于背景的部分(白色线，可以画多条)\n' \
            '最终点击鼠标中间滚轮开始执行裁剪算法；\n\n' + \
            'Final: 匹配结果可以点击Show Result查看！！！'
        tkinter.messagebox.showinfo('使用说明', s)

    def begin_grabcut(self, src_w):
        src_w.destroy()
        demo_window = tkinter.Tk()
        self.src_w = demo_window
        demo_window.protocol("WM_DELETE_WINDOW", self.on_exit)
        # ###重写×控件####
        screenwidth = demo_window.winfo_screenwidth()
        screenheight = demo_window.winfo_screenheight()
        demo_window.geometry(
            '%dx%d+%d+%d' % (700, 400, (screenwidth - 500 - 150) / 2, (screenheight - 400 - 150) / 2))
        demo_window.title('Grabcut')

        ### Source Imaeg 原图存放 ###
        s_image = tkinter.StringVar(demo_window)
        s_image.set('Source Image')
        src_label = tkinter.Label(textvariable=s_image, fg='black', font=('Times', 15))
        src_label.place(x=140, y=150)
        photo, _, _ = get_img_obj('resources/11.jpg', self.first_width)
        photo = ImageTk.PhotoImage(image=photo)
        img_label = tkinter.Label(demo_window, imag=photo)
        img_label.place(x=50, y=200)

        ### Target Imaeg 目标图存放 ###
        t_image = tkinter.StringVar(demo_window)
        t_image.set('Target Image')
        tar_label = tkinter.Label(textvariable=t_image, fg='black', font=('Times', 15))
        tar_label.place(x=450, y=150)
        t_label = tkinter.Label(demo_window, imag=photo)
        t_label.place(x=350, y=200)

        choose_src = tkinter.Button(demo_window, text='Choose Src Image', fg='black', font=('Times', 12))
        choose_src.bind('<Button-1>', lambda event: self.get_image(img_label, s_image))
        choose_src.place(x=130, y=110)

        choose_tar = tkinter.Button(demo_window, text='Show Result', fg='black', font=('Times', 12))
        choose_tar.bind('<Button-1>', lambda event: self.show())
        choose_tar.place(x=453, y=110)

        begin = tkinter.Button(demo_window, text='Begin GrabCut', fg='black', font=('Times', 12))
        begin.bind('<Button-1>', lambda event: self.begin(s_pbar, p1, t_label))
        begin.place(x=300, y=110)

        ### 进度条 ###
        p1 = ttk.Progressbar(demo_window, length=600, mode='determinate', orient=HORIZONTAL)
        p1.place(x=50, y=65)
        p1['maximum'] = 100
        p1['value'] = 0
        ### 显示进度条的信息 ###
        s_pbar = tkinter.StringVar(demo_window)
        s_pbar.set('Progress Will Be Shown Here...')
        pbar_label = tkinter.Label(demo_window, textvariable=s_pbar, font=('Times', 12))
        pbar_label.place(x=250, y=20)

        ##########主菜单是menu##################
        menu = tkinter.Menu(demo_window)
        demo_window.config(menu=menu)
        first_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='GrabCut使用说明', menu=first_menu)
        first_menu.add_command(label='GrabCut使用说明', command=self.how_to_use)
        second_menu = tkinter.Menu(menu, tearoff=False)
        menu.add_cascade(label='开发者', menu=second_menu)
        second_menu.add_command(label='关于开发者', command=developer)

        demo_window.mainloop()

def developer():
    s = 'Made By Nankai University:\n  计算机学院：王博源'
    tkinter.messagebox.showinfo('关于开发者！', s)

def clear():
    list_dirs = os.listdir('resources')
    for d in list_dirs:
        d1 = os.path.join('resources', d)
        if not os.path.isdir(d1): continue
        files = os.listdir(d1)
        for f in files:
            p = os.path.join(d1, f)
            if os.path.isfile(p):
                os.remove(p)
    tkinter.messagebox.showinfo('警告！', '已经全部删除！')

def window_main():
    demo_window = tkinter.Tk()
    # ###重写×控件####
    screenwidth = demo_window.winfo_screenwidth()
    screenheight = demo_window.winfo_screenheight()
    demo_window.geometry(
        '%dx%d+%d+%d' % (500, 350, (screenwidth - 500 - 150) / 2, (screenheight - 350 - 150) / 2))
    demo_window.title('Image Processing')

    patchmatch_ui = PatchMatch_UI()
    btn1 = tkinter.Button(demo_window, text='Essay1: Patch Match', fg='black', font=('Times',12),
                          width = 18, height = 3)
    btn1.bind('<Button-1>', lambda event: patchmatch_ui.begin_patchmatch(demo_window))
    btn1.place(x = 50, y = 40)

    seamcarve_ui = SeamCarve_UI()
    btn2 = tkinter.Button(demo_window, text='Essay2: Seam Carving', fg='black', font=('Times', 12),
                          width=18, height=3)
    btn2.bind('<Button-1>', lambda event: seamcarve_ui.begin_seamcarve(demo_window))
    btn2.place(x = 50, y=140)

    imagewarp_ui = ImageWarp_UI()
    btn3 = tkinter.Button(demo_window, text='Essay3: Image Warping', fg='black', font=('Times', 12),
                          width=18, height=3)
    btn3.bind('<Button-1>', lambda event: imagewarp_ui.begin_warp(demo_window))
    btn3.place(x=50, y=240)

    wls_ui = WLSFilter_UI()
    btn4 = tkinter.Button(demo_window, text='Essay4: WLS Filter', fg='black', font=('Times', 12),
                          width=18, height=3)
    btn4.bind('<Button-1>', lambda event: wls_ui.begin_wls(demo_window))
    btn4.place(x=250, y=40)

    decomp_ui = Decomposition_UI()
    btn5 = tkinter.Button(demo_window, text='Essay5: Edge-Preserving', fg='black', font=('Times', 12),
                          width=18, height=3)
    btn5.bind('<Button-1>', lambda event: decomp_ui.begin_decomposition(demo_window))
    btn5.place(x=250, y=140)

    grabcut_ui = GrabCut_UI()
    btn6 = tkinter.Button(demo_window, text='Essay6: Grab Cut', fg='black', font=('Times', 12),
                          width=18, height=3)
    btn6.bind('<Button-1>', lambda event: grabcut_ui.begin_grabcut(demo_window))
    btn6.place(x=250, y=240)

    #########下面加一些菜单#########################
    menu = tkinter.Menu(demo_window)
    demo_window.config(menu=menu)
    ##########主菜单是menu##################
    first_menu = tkinter.Menu(menu, tearoff=False)
    menu.add_cascade(label='开发者', menu=first_menu)
    first_menu.add_command(label='关于开发者', command=developer)
    second_menu = tkinter.Menu(menu, tearoff=False)
    menu.add_cascade(label='Clear', menu=second_menu)
    second_menu.add_command(label='Clear All', command=clear)
    demo_window.mainloop()


if __name__ == '__main__':
    window_main()