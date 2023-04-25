from CBMIR_tensorboard import CBMIR
import tkinter as tk
import  os,h5py,shutil
from tkinter import filedialog


root = tk.Tk()

w=400+200  #width
r=400+400  #height
x=550  #與視窗左上x的距離
y=90  #與視窗左上y的距離
root.geometry('%dx%d+%d+%d' % (w,r,x,y))
target_dataset = []
query_dataset = []



def choose_folder():
    # 讓使用者選擇資料夾
    folder_path = filedialog.askdirectory()
    if folder_path:
        # 使用os模組讀取資料夾中的所有子目錄
        subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]

        # 創建Checkbutton
        checkbox_vars = []
        checkbox_list = []
        for folder_name in subfolders:
            var = tk.IntVar()
            checkbox_vars.append(var)
            checkbutton = tk.Checkbutton(save_group, text=folder_name, variable=var)
            checkbutton.pack()
            checkbox_list.append(checkbutton)

        # 創建印出勾選值的按鈕
        def print_selected_values():
            selected_values = [subfolders[i] for i in range(len(subfolders)) if checkbox_vars[i].get() == 1]
            target_dataset = selected_values
            print("勾選的值：",target_dataset,type(target_dataset))
            if  os.path.exists('target_data'):
                shutil.rmtree('target_data')

            for i in os.listdir('data'):
                for j in os.listdir('data/'+i):
                    if j not in target_dataset:
                        continue
                    for k in os.listdir('data/'+i+'/'+j):
                        if not os.path.exists('target_data/'+i+'/'+j):
                            os.makedirs('target_data/'+i+'/'+j)
                        shutil.copy(('data/'+i+'/'+j+'/'+k),('target_data/'+i+'/'+j+'/'+k))


            target_mylabel = tk.Label(save_group, text='target ok')
            target_mylabel.pack()
            # 關閉Checkbutton
            for checkbox in checkbox_list:
                checkbox.pack_forget()
                print_button.pack_forget()

        print_button = tk.Button(save_group, text="印出勾選的值", command=print_selected_values)
        print_button.pack()


def choose_folder2():
    # 讓使用者選擇資料夾
    folder_path = filedialog.askdirectory()
    if folder_path:
        # 使用os模組讀取資料夾中的所有子目錄
        subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]

        # 創建Checkbutton
        checkbox_vars = []
        checkbox_list = []
        for folder_name in subfolders:
            var = tk.IntVar()
            checkbox_vars.append(var)
            checkbutton = tk.Checkbutton(save_group, text=folder_name, variable=var)
            checkbutton.pack()
            checkbox_list.append(checkbutton)

        # 創建印出勾選值的按鈕
        def print_selected_values():
            selected_values = [subfolders[i] for i in range(len(subfolders)) if checkbox_vars[i].get() == 1]
            query_dataset = selected_values
            print("勾選的值：",query_dataset,type(query_dataset))

            if  os.path.exists('query_data'):
                 shutil.rmtree('query_data')

            for i in os.listdir('data'):
                for j in os.listdir('data/'+i):
                    if j not in query_dataset:
                        continue
                    for k in os.listdir('data/'+i+'/'+j):
                        if not os.path.exists('query_data/'+i+'/'+j):
                            os.makedirs('query_data/'+i+'/'+j)
                        shutil.copy(('data/'+i+'/'+j+'/'+k),('query_data/'+i+'/'+j+'/'+k))

            query_mylabel = tk.Label(save_group, text='query ok')
            query_mylabel.pack()
            # 關閉Checkbutton
            for checkbox in checkbox_list:
                checkbox.pack_forget()
                print_button.pack_forget()

        print_button = tk.Button(save_group, text="印出勾選的值", command=print_selected_values)
        print_button.pack()


def train(rep = 0):
    cb = CBMIR()
    cb.data_path = data_path.get()
    cb.save_path=save_folder.get() + "_"+str(rep)+'\\'


    cb.model_listt = []
    if (swinVar1.get())==1:
        cb.model_listt.append('swin_vit')
    if(vitVar1.get()==1):
        cb.model_listt.append('vit')
    if(densenetVar1.get()==1):
        cb.model_listt.append('densenet')
    
    if path.get() == 'all':
        cb.path_listt = []
        for i in os.listdir(data_path.get()):
            cb.path_listt.append(i)
    else:
        cb.path_listt = []
        for i in path.get() :
            if i == ',':
                continue
            else:
                cb.path_listt.append(i)
     
    
    cb.train_typee = []
    if(scratch.get()==1):
        cb.train_typee.append('trainFromScratch')
    if(Finetune.get()==1):
        cb.train_typee.append('finetune')
    cb.num_epochs = int(epoch1.get())
    cb.batch_size = int(batch_size.get())
    cb.K = int(k.get())
    print(cb.data_path,cb.save_path,cb.model_listt,cb.train_typee,cb.K,cb.batch_size,cb.num_epochs,cb.path_listt)
    #cb.foldd(cb.data_path)
    cb.data_path = 'input_data'
    cb.auto_train()
    

def split():
    cb = CBMIR()
    cb.K = int(k.get())
    cb.data_path = str(data_path.get())
    cb.save_path=str(save_folder.get())
    cb.foldd(cb.data_path)


def retireve(rep = 0):
    cb = CBMIR()
    cb.data_path = data_path.get()
    #cb.save_path=save_folder.get()
    cb.save_path=save_folder.get() + "_"+str(rep)+'\\'
    cb.model_listt = []
    if (swinVar1.get())==1:
        cb.model_listt.append('swin_vit')
    if(vitVar1.get()==1):
        cb.model_listt.append('vit')
    if(densenetVar1.get()==1):
        cb.model_listt.append('densenet')
    if path.get() == 'all':
        cb.path_listt = []
        for i in os.listdir(data_path.get()):
            cb.path_listt.append(i)
    else:
        cb.path_listt = []
        for i in path.get() :
            if i == ',':
                continue
            else:
                cb.path_listt.append(i)

    cb.train_typee = []
    if(scratch.get()==1):
        cb.train_typee.append('trainFromScratch')
    if(Finetune.get()==1):
        cb.train_typee.append('finetune')
    cb.num_epochs = int(epoch1.get())
    cb.batch_size = int(batch_size.get())
    cb.K = int(k.get())
    print(cb.data_path,cb.save_path,cb.model_listt,cb.train_typee,cb.K,cb.batch_size,cb.num_epochs,cb.path_listt)
    #cb.foldd(cb.data_path)
    cb.data_path = 'input_data'
    #cb.auto_train()
    cb.retireve()


def onlt_get_feature():
    cb = CBMIR()
    cb.save_path=save_folder.get()
    cb.model_listt = []
    if (swinVar1.get())==1:
        cb.model_listt.append('swin_vit')
    if(vitVar1.get()==1):
        cb.model_listt.append('vit')
    if(densenetVar1.get()==1):
        cb.model_listt.append('densenet')
    if path.get() == 'all':
        cb.path_listt = []
        for i in os.listdir(data_path.get()):
            cb.path_listt.append(i)
    else:
        cb.path_listt = []
        for i in path.get() :
            if i == ',':
                continue
            else:
                cb.path_listt.append(i)
   
    cb.K = int(k.get())
    print(cb.data_path,cb.save_path,cb.model_listt,cb.train_typee,cb.K,cb.batch_size,cb.num_epochs,cb.path_listt)
    cb.pretrain_retireve()


def run():
    if (splitt.get())==1:
        split()
    if(traintrain.get()) == 1:
        for ii in range(int(kk.get())):
            train(rep=ii)
    if(get_feature.get()) == 1:
        for ii in range(int(kk.get())):
            retireve(rep=ii)
    if(onlt_get_featureonlt_get_feature.get()) == 1:
        onlt_get_feature()
    

group = tk.LabelFrame(root,  padx=20, pady=20)
group.pack()

save_group = tk.LabelFrame(root, padx=20, pady=20)
save_group.pack()

parm_group = tk.LabelFrame(root,  padx=20, pady=20)
parm_group.pack()
#############################################
#選擇模型訓練 vit or densenet or swin_vit
##############################################    
mylabel = tk.Label(group, text='選擇模型：')
mylabel.pack()

swinVar1 = tk.IntVar()
vitVar1 = tk.IntVar()
densenetVar1 = tk.IntVar()
C1 = tk.Checkbutton(group, text = "swin", variable = swinVar1)
C2 = tk.Checkbutton(group, text = "vit", variable = vitVar1)
C3 = tk.Checkbutton(group, text = "densenet", variable = densenetVar1)

C1.pack()
C2.pack()
C3.pack()
###############################################
#訓練方式
#############################################
mylabel = tk.Label(parm_group, text='訓練方式')
mylabel.pack()

scratch = tk.IntVar()
Finetune = tk.IntVar()
C11 = tk.Checkbutton(parm_group, text = "train from scratch", variable = scratch)
C22 = tk.Checkbutton(parm_group, text = "finetune", variable = Finetune)

C11.pack()
C22.pack()

# #############################################
# #資料集路徑
# #############################################
# mylabel = tk.Label(save_group, text='資料集路徑')
# mylabel.pack()
data_path = tk.StringVar(value='data')
# data_path = tk.Entry(save_group,textvariable=data_path)
#data_path.pack()

#############################################
#存檔路徑
#############################################
mylabel = tk.Label(save_group, text='存檔路徑')
mylabel.pack()
save_folder = tk.StringVar(value='save')
save_folder = tk.Entry(save_group,textvariable=save_folder)
save_folder.pack()
#############################################
#系列
#############################################
# mylabel = tk.Label(save_group, text='來源')
# mylabel.pack()
path = tk.StringVar(value='all')
#path = tk.Entry(save_group,textvariable=path)
#path.pack()

#############################################
#存檔路徑
#############################################
# mylabel = tk.Label(save_group, text='目標影像資料集路徑')
# mylabel.pack()
# save_folder1 = tk.StringVar(value='target_data')
# save_folder1 = tk.Entry(save_group,textvariable=save_folder1)
# save_folder1.pack()

# 創建啟動選擇資料夾功能的按鈕
start_button = tk.Button(save_group, text="選擇目標影像資料集", command=choose_folder)
start_button.pack()

start_button = tk.Button(save_group, text="選擇目標影像資料集", command=choose_folder2)
start_button.pack()

# retrieve_btn = tk.Button(save_group,text="確定檢索資料 ",command=retrieve_data)
# retrieve_btn.pack(pady=10, side='bottom')
# #############################################
# #存檔路徑
# #############################################
# mylabel = tk.Label(save_group, text='查詢影像資料集路徑')
# mylabel.pack()
# save_folder2 = tk.StringVar(value='query_data')
# save_folder2 = tk.Entry(save_group,textvariable=save_folder2)
# save_folder2.pack()

#############################################
#迭代次數
#############################################
mylabel = tk.Label(parm_group, text='迭代次數')
mylabel.pack()
epoch = tk.IntVar(value=20)
epoch1 = tk.Entry(parm_group,textvariable=epoch)
epoch1.pack()

#############################################
#批次大小 
#############################################

mylabel = tk.Label(parm_group, text='批次大小')
mylabel.pack()
batch_size = tk.StringVar(value=16)
batch_size = tk.Entry(parm_group,textvariable=batch_size)
batch_size.pack()

#############################################
#交叉驗證數量
#############################################

mylabel = tk.Label(parm_group, text='交叉驗證數量')
mylabel.pack()
k = tk.StringVar(value=2)
k = tk.Entry(parm_group,textvariable=k)
k.pack()

#############################################
#執行次數
#############################################

mylabel1 = tk.Label(parm_group, text='執行次數')
mylabel1.pack()
kk = tk.StringVar(value=1)
kk= tk.Entry(parm_group,textvariable=kk)
kk.pack()


mylabel = tk.Label(parm_group, text='選擇功能：')
mylabel.pack()

splitt = tk.IntVar()
traintrain = tk.IntVar()
get_feature = tk.IntVar()
onlt_get_featureonlt_get_feature = tk.IntVar()
C111 = tk.Checkbutton(parm_group, text = "split", variable = splitt)
C222 = tk.Checkbutton(parm_group, text = "train", variable = traintrain)
C333 = tk.Checkbutton(parm_group, text = "retireve", variable = get_feature)
C444 = tk.Checkbutton(parm_group, text = "pretrain_retireve", variable = onlt_get_featureonlt_get_feature)


C111.pack()
C222.pack()
C333.pack()
C444.pack()

btn0 = tk.Button(parm_group,text=" run ",command=run)
btn0.pack(pady=10)
root.mainloop()
