import h5py
import os,csv
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import h5py

root = tk.Tk()
# 設置全屏
root.attributes('-fullscreen', 1)
button = tk.Button(root, text='Close', command=root.destroy, font=('Arial',14), width=10, height=2)

# button = tk.Button(root, text='Close', command=root.destroy)
button.place(relx=1.0, y=0, anchor='ne')

# 创建窗口

group = tk.LabelFrame(root,  padx=20, pady=20)
group.pack()

save_group = tk.LabelFrame(root, padx=20, pady=20)
save_group.pack()

#############################################
#存檔路徑
#############################################
mylabel = tk.Label(save_group, text='存檔路徑')
mylabel.pack(side=tk.LEFT)

save_folder = tk.StringVar(value='retrieve_test')
save_folder = tk.Entry(save_group,textvariable=save_folder)
save_folder.pack(side=tk.LEFT)

mylabe0 = tk.Label(save_group, text='  |  \n  |  \n  |  \n  |  ')
mylabe0.pack(side=tk.LEFT)

def choose_folder():
    folder_path = filedialog.askdirectory()
    print("所选文件夹路径：", folder_path)
    global save_path
    save_path.set(folder_path)

save_path = tk.StringVar()
# 將var對象與Entry部件綁定
# entry = tk.Label(root, textvariable=save_path)
# entry.pack()

choose_folder_button = tk.Button(save_group, text="選擇資料夾", command=choose_folder)
choose_folder_button.pack(side=tk.LEFT)


mylabe0 = tk.Label(save_group, text='  |  \n  |  \n  |  \n  |  ')
mylabe0.pack(side=tk.LEFT)
#############################################
#選擇模型訓練 vit or densenet or swin_vit
##############################################    
mylabel = tk.Label(save_group, text='選擇模型：')
mylabel.pack(side=tk.LEFT)

model_Var1 = tk.StringVar()


C1 = tk.Radiobutton(save_group, text = "swin", variable = model_Var1, value="swin")
C2 = tk.Radiobutton(save_group, text = "vit", variable = model_Var1, value="vit")
C3 = tk.Radiobutton(save_group, text = "densenet", variable = model_Var1, value="densenet")

C1.pack(side=tk.LEFT)
C2.pack(side=tk.LEFT)
C3.pack(side=tk.LEFT)
# # 將var對象與Entry部件綁定
# mylabe0 = tk.Label(root, textvariable=model_Var1)
# mylabe0.pack()
mylabe0 = tk.Label(save_group, text='  |  \n  |  \n  |  \n  |  ')
mylabe0.pack(side=tk.LEFT)

#############################################
#選擇模型訓練 vit or densenet or swin_vit
##############################################    
mylabel = tk.Label(save_group, text='選擇訓練模式：')
mylabel.pack(side=tk.LEFT)

model_Var2 = tk.StringVar()


C1 = tk.Radiobutton(save_group, text = "train from scratch", variable = model_Var2, value="train from scratch")
C2 = tk.Radiobutton(save_group, text = "finetune", variable = model_Var2, value="finetune")


C1.pack(side=tk.LEFT)
C2.pack(side=tk.LEFT)

# # 將var對象與Entry部件綁定
# mylabe0 = tk.Label(root, textvariable=model_Var2)
# mylabe0.pack()

mylabe0 = tk.Label(save_group, text='  |  \n  |  \n  |  \n  |  ')
mylabe0.pack(side=tk.LEFT)
#############################################
#fold
#############################################
mylabel2 = tk.Label(save_group, text='fold')
mylabel2.pack(side=tk.LEFT)

save_folder2 = tk.StringVar(value='0')
save_folder2 = tk.Entry(save_group,textvariable=save_folder2)
save_folder2.pack(side=tk.LEFT)
mylabe0 = tk.Label(save_group, text='  |  \n  |  \n  |  \n  |  ')
mylabe0.pack(side=tk.LEFT)
# # 將var對象與Entry部件綁定
# mylabe0 = tk.Label(root, textvariable=mylabe0)
# mylabe0.pack()

#############################################
#top
#############################################
mylabel3 = tk.Label(save_group, text='top')
mylabel3.pack(side=tk.LEFT)

topp = tk.IntVar(value=10)
topp = tk.Entry(save_group,textvariable=topp)
topp.pack(side=tk.LEFT)
# 將var對象與Entry部件綁定
# mylabe0 = tk.Label(root, textvariable=topp)
# mylabe0.pack()

def open_image():
    
    def densenet_feature(img_path):


                transform = transforms.Compose([
                        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
                ]) 

                img = Image.open(img_path)
                img = transform(img)
                img = torch.unsqueeze(img, dim=0)
                CUDA = torch.cuda.is_available()
                device = torch.device("cpu" if CUDA else "cpu")
                img = img.to(device)

                # create model
                
            
                #a = model.features.children
                
                # model=torch.load('S_8_0\\finetune\S\densenet\\2.pth')
                with torch.no_grad():
                    input = model.features(img)
                    avgPooll = nn.AdaptiveAvgPool2d(1)

                    output = avgPooll(input)
                    output = torch.transpose(output, 1, 3)#把通道维放到最后
                    featuree = output.view(1920).cpu().numpy()
                    print(featuree.shape)
                    return featuree


    def swin_feature(img_path):


                    transform = transforms.Compose([
                            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
                    ]) 

                    img = Image.open(img_path)
                    img = transform(img)
                    img = torch.unsqueeze(img, dim=0)
                    CUDA = torch.cuda.is_available()
                    device = torch.device("cpu" if CUDA else "cpu")
                    img = img#.to(device)

                    # create model
                    #model = torch.load('swin_b-68c6b09e.pth')
                    model = torchvision.models.swin_b(weights = None)
                    #model.load_state_dict(torch.load('swin_b-68c6b09e.pth'))
                    #model = torchvision.models.swin_b(weights = 1)
                    #model.cuda()
                    model.eval()


                    with torch.no_grad():
                        #https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029/2
                        input = model.features(img)
                        #7,7,1024 -> 1,1,1024
                        avgPooll = nn.AdaptiveAvgPool2d(1)
                        input = torch.transpose(input, 1, 3)#把通道维放到最后
                        output = avgPooll(input)

                        #swin b 1024 features 1,1,1024-> 1024
                        featuree = output.view(1024).cpu().numpy()
                        return featuree


    def vit_feature(img_path):
            '''
            extract feature from an image
            '''
            # image ->(3,224,224)
            transform = transforms.Compose([
                    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
            ])

            # use path open image
           
            img = Image.open(img_path)
            print(img_path, img_path[:-3])
            img = transform(img)

            #通道多一條
            img = torch.unsqueeze(img, dim=0)
            
            CUDA = torch.cuda.is_available()
            device = torch.device("cpu" if CUDA else "cpu")
            img = img.to(device)
            #print(img.shape)

            # create model
            #model = torch.load(model_path)
            #model = torchvision.models.vit_b_16(weights = None) 
            # model = torch.load('S_8_0\\finetune\S\\vit\\2.pth')
            #model.cuda()
            model.eval()
            
            
            with torch.no_grad():
                x = model._process_input(img)
                n = x.shape[0]
                batch_class_token = model.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = model.encoder(x)
                x = x[:, 0]
                featuree = x.view(768).cpu().numpy()
                #print(featuree.shape)
                return featuree


    # 打开对话框，让用户选择图片文件
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    
    # # 如果用户选择了文件，将文件路径传递给 Image 对象，并缩放图像
    # if file_path:
    #     image_file = Image.open(file_path)
    #     resized_image = image_file.resize((200, 150))
        
    #     # 将缩放后的图像转换为 PhotoImage 对象，并显示在标签上
    #     photo_image = ImageTk.PhotoImage(resized_image)
    #     label.configure(image=photo_image)
    #     label.image = photo_image
    # print(file_path)
    
    path = file_path
    global save_path
    # print(save_path.get())

    print(save_path,type(save_path))
    a = (save_path.get()) 
    b = (model_Var2.get())
    model_path = a + '/' + b
    print(model_path)
    for i in os.listdir(model_path):
          model_path = model_path  + '/' + i
          print(i)
    model_path = model_path  + '/' + model_Var1.get()+ '/' +(save_folder2.get() )+ '.pth'
    print(model_path)
    feature_h5 = ''


    device = torch.device('cpu')
    model = torch.load((model_path), map_location=device)
  

    feature_list = []
    feature_path = []

    dataa = "finetune_retrieve" if model_Var2.get() == 'finetune' else 'retrieve_trainFromScratch'
    feature_h5 = save_path.get() + '/' +dataa
    for i in os.listdir(feature_h5):
          feature_h5 = feature_h5  + '/' + i
    print(feature_h5)
    feature_h5 = feature_h5  + '/' + model_Var1.get()+ '/fold' +str(save_folder2.get() )+ '/target_data.h5'

    # feature_h5 = "S_8_0\\finetune_retrieve\S\\vit\\fold0\\target_data.h5"
    with h5py.File(feature_h5, "r") as k:
                for i in range(len(k.get('feature'))):
                    feature_list.append(k.get('feature')[i])
                    feature_path.append(k.get('path')[i])
                #查詢影像
    query  = (vit_feature(path)) 
    query_path = path

    #用來存放每張圖的cosin similarity
    score_map = {}

    #比對資料庫中的每一筆DATA
    for i in range(len(feature_list)):

        #計算cosin similarity
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        #需要轉tensor
        cosine_similarity = cos(torch.from_numpy(query),torch.from_numpy(feature_list[i]))
        #寫入 dic
        score_map.update({cosine_similarity:feature_path[i]})
    #將前 n 相似的輸出
    top_list = {}
    image = []
    image.append(path)
    print(image)
    #top_10
    top = topp.get()
    top = int(top)
    for i in range(top):
        #每次都挑最大的放入dic 概念類似選擇排序
        top_list.update({max(score_map):score_map[max(score_map)]})
        #將最大的刪除
        del score_map[max(score_map)]
    #print(top_list)
    relevant = 0 
    #印出top-n串列
    for i in top_list:
        print(query_path,(str(top_list[i])))
        if (str(query_path).split('/')[-2]) == (str(top_list[i]).split('\\')[-3]):
            relevant = relevant + 1

        print(top_list[i])
        image.append(str((top_list[i]))[2:-1]+'*'+str(float(i))[:4])
        #print(image)
        #print(aaa)
    ap = relevant/top
    print(query_path)
    print('ap:',ap)
    with open('result/swin_vit_output.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([path,ap])
    print(len(feature_list))
    #path = path.split('\\')[1]+path.split('\\')[2]+'_'+path.split('\\')[3]
    fig, axs = plt.subplots(4, 4, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    axs = axs.flatten()  # 将子图数组展开为一维数组
    for i, image_path in enumerate(image):
        # 读取图像
        with open(image_path.split('*')[0], 'rb') as f:
            img = Image.open(f)
            # 显示图像和标题
            axs[i].imshow(img)
            axs[i].get_xaxis().set_visible(False)  
            axs[i].get_yaxis().set_visible(False)
            path=path
            if i == 0:
                axs[i].set_title(f"{'query'} ")
            else:
                a = (image_path.split('*')[0]).split('\\')[4]
                axs[i].set_title(f"{i}\n {os.path.basename(image_path.split('*')[1])}\nS{a}")
            f.close()

    # 移除多余的子图
    for i in range(len(image), len(axs)):
        axs[i].remove()
    if not os.path.exists(save_folder.get()):
                os.makedirs(save_folder.get())
    result_save_path = save_folder.get() + '/'+(file_path.split('/')[-2])+"_"+(file_path.split('/')[-1]).split('.')[0]+'_top_'+str(topp.get())+'.png'       
    plt.savefig(result_save_path)
    #plt.show()
    plt.clf()
    plt.close('all')
    file_path = result_save_path
    image_file = Image.open(file_path)
    #resized_image = image_file.resize((900, 600))
    
    # 将缩放后的图像转换为 PhotoImage 对象，并显示在标签上
    photo_image = ImageTk.PhotoImage(image_file)
    label1.configure(image=photo_image)
    label1.image = photo_image

# 创建打开按钮
open_button = tk.Button(root, text="Open", command=open_image)
open_button.pack()

# 创建标签
label1 = tk.Label(root)
label1.pack()
# 运行窗口主循环
root.mainloop()
