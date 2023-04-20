import sys
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import images
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd 

np.set_printoptions(threshold=sys.maxsize)
GambarLs=[336,337,338]
labels=["336","337","338"]
second=["happy","neutral","sad"]


class Main_Frame():
    def __init__(self) -> None:
        self.win=tk.Tk()
        self.win.geometry("500x500")
        self.main_frame=tk.Frame(self.win)
        self.main_frame.pack(fill=tk.BOTH,expand=1)

        self.main_canvas=tk.Canvas(self.main_frame)
        self.main_canvas.pack(side=tk.LEFT,expand=1,fill=tk.BOTH)
        # self.main_canvas.grid(row=0,column=0,sticky="nsew")

        self.MyscrollbarVertical=tk.Scrollbar(self.main_frame,orient=tk.VERTICAL,command=self.main_canvas.yview)
        # self.MyscrollbarVertical.pack(side=tk.RIGHT,fill=tk.Y)
        # self.MyscrollbarVertical.grid(row=0,column=1,rowspan=2,sticky="ns")
        self.MyscrollbarVertical.place(x=0,y=0,height=200)
        self.MyscrollbarHorizontal=tk.Scrollbar(self.main_frame,orient=tk.HORIZONTAL,command=self.main_canvas.xview)
        # self.MyscrollbarHorizontal.pack(side=tk.BOTTOM,fill=tk.X)
        # self.MyscrollbarHorizontal.grid(row=1,column=0,columnspan=2,sticky="ew")
        self.MyscrollbarHorizontal.place(x=20,y=0,width=200)

        self.main_canvas.configure(yscrollcommand=self.MyscrollbarVertical.set)
        self.main_canvas.configure(xscrollcommand=self.MyscrollbarHorizontal.set)
        self.main_canvas.bind('<Configure>',lambda e:self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))
        self.second_frame=tk.Frame(self.main_canvas)
        self.newframe=tk.Frame(self.second_frame)
        self.newframe.pack(pady=20)
        self.numberinput=tk.Entry(self.newframe)
        label=tk.Label(self.newframe,{"text":"masukan input angka:"})
        label.grid(row=0,column=1)
        self.numberinput.grid(row=1,column=1)
        self.button_sumbit=tk.Button(self.newframe,{"text":"Sumbit"},command=self.retive_data)
        self.button_sumbit.grid(row=2,column=0)
        self.button_color=tk.Button(self.newframe,{"text":"Colour analysis"},command=self.color_analysis)
        self.button_color.grid(row=2,column=1)
        self.button_texture=tk.Button(self.newframe,{"text":"Texture analysis"},command=self.texture_analysis)
        self.button_texture.grid(row=2,column=2)
        self.answer=tk.Label(self.second_frame)
        self.answer.pack()
        self.answer["text"]=""
        self.second_frame.pack()
    def inialize(self):
        self.win.mainloop()
    def retive_data(self):
        self.data=images.Readnumber(int(self.numberinput.get()))
        self.GLCM=images.FromListToGlcm(self.data)
        self.mean=[]
        for i in self.data:
            self.mean.append(np.mean(i))
        self.std=[]
        for i in self.data:
            self.std.append(np.std(i))
        self.contrast_feature=[]
        for i in self.GLCM:
            self.contrast_feature.append(images.contrast_feature(i))
        self.dissimilarity_feature=[]
        for i in self.GLCM:
            self.dissimilarity_feature.append(images.dissimilarity_feature(i))
        self.homogeneity_feature=[]
        for i in self.GLCM:
            self.homogeneity_feature.append(images.homogeneity_feature(i))
        self.energy_feature=[]
        for i in self.GLCM:
            self.energy_feature.append(images.energy_feature(i))
        self.correlation_feature=[]
        for i in self.GLCM:
            self.correlation_feature.append(images.correlation_feature(i))
        self.asm_feature=[]
        for i in self.GLCM:
            self.asm_feature.append(images.asm_feature(i))
        self.answer["text"]=""
    
        
    def color_analysis(self):
        fig ,axis=plt.subplots(2,3)
        fig.tight_layout(pad=0.5)
        fig.set_figheight(8)
        fig.set_figwidth(10)
        axis=axis.flatten()
        for i in range(3):
            bins=np.linspace(np.min(self.data[i]),np.max(self.data[i]),10)
            plt.sca(axis[i])
            plt.title(second[i])
            plt.hist(self.data[i],bins=bins)
            plt.sca(axis[i+3])
            plt.imshow(self.data[i])
        thetext=""
        for i in range(3):
            thetext+="{} \nmean : {} \nstandardenviation : {}\n".format(second[i],self.mean[i],self.std[i])
        self.answer["text"]=thetext
        plt.show()

    def texture_analysis(self):
        fig ,axis=plt.subplots(4,3)
        fig.tight_layout(pad=0.5)
        fig.set_figheight(8)
        fig.set_figwidth(10)
        axis=axis.flatten()
        for i in range(len(axis)):
            plt.sca(axis[i])
            plt.title("{} degree : {}".format(second[i%3],i//3))
            plt.imshow(self.GLCM[i%3][:,:,0,i//3])
        thetext=""
        for i in range(3):
            for z in range(4):
                thetext+="{} \nside : {} \ncontrast_feature : {} \ndissimilarity_feature : {} \nhomogeneity_feature : {} \nenergy_feature: {} \nasm_feature:{}\n".format(second[i],z+1,self.contrast_feature[i][0,z],self.dissimilarity_feature[i][0,z],self.homogeneity_feature[i][0,z],self.energy_feature[i][0,z],self.asm_feature[i][0,z])
        self.answer["text"]=thetext
        plt.show()
        
     
    
frame=Main_Frame()
frame.inialize()

