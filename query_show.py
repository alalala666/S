import h5py
import os,csv
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
def densenet_feature(img_path):


            transform = transforms.Compose([
                    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
            ]) 

            img = Image.open(img_path)
            img = transform(img)
            img = torch.unsqueeze(img, dim=0)
            CUDA = torch.cuda.is_available()
            device = torch.device("cuda" if CUDA else "cpu")
            img = img.to(device)

            # create model
            
           
            #a = model.features.children
            
            # model=torch.load('S_8_1\\finetune\S\densenet\\2.pth')
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
                device = torch.device("cuda" if CUDA else "cpu")
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
            device = torch.device("cuda" if CUDA else "cpu")
            img = img.to(device)
            #print(img.shape)

            # create model
            #model = torch.load(model_path)
            #model = torchvision.models.vit_b_16(weights = None) 
            # model = torch.load('S_8_1\\finetune\S\\vit\\2.pth')
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


#----------------------------------------------------------
#              input feature and path
#----------------------------------------------------------

ap=0
feature_list = []
feature_path = []
use_model = 'densenet'
count = 0
#model = torchvision.models.densenet_b_16(weights = None) 
model = torch.load('S_8_1\\finetune\S\\densenet\\0.pth')
for file_class in os.listdir('query_data\S'):
#   #continue
#   if count == 0:
#          count += 1
#          continue
    #try:    
        for filename in os.listdir('query_data\S' +"\\"+ file_class):
            
            path = 'query_data\S' +"\\"+ file_class + '\\'+filename
            # print()
            # continue
            feature_list = []
            feature_path = []


            feature_h5 = "S_8_1\\finetune_retrieve\S\\densenet\\fold0\\target_data.h5" 
            with h5py.File(feature_h5, "r") as k:
                for i in range(len(k.get('feature'))):
                    feature_list.append(k.get('feature')[i])
                    feature_path.append(k.get('path')[i])
            

            #----------------------------------------------------------
            #               mAP compute
            #----------------------------------------------------------

            #將結果寫入csv
            #先宣告要用的欄位
            # ap , 分類 , 路徑
        

            #查詢影像
            query  = (densenet_feature(path)) #if use_model == 'vit'else (swin_feature(path))
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
            ralavent_target = []

            
            #top_10
            top = 10
            for i in range(top):
                #每次都挑最大的放入dic 概念類似選擇排序
                top_list.update({max(score_map):score_map[max(score_map)]})
                #將最大的刪除
                del score_map[max(score_map)]
            #print(top_list)
            relevant = 0 
            #印出top-n串列
            for i in top_list:
                if (str(query_path).split('\\')[-2]) == (str(top_list[i]).split('\\')[-3]):
                    relevant = relevant + 1

                print(top_list[i])
                image.append(str((top_list[i]))[2:-1]+'*'+str(float(i))[:4])
                print(((str((top_list[i]))[2:-1])))
                ralavent_target.append((str((top_list[i]))[2:-1]).split('S_')[1].split('_')[0])
               # print('ralavent_target : ',ralavent_target)
                
                #print(image)
                #print(aaa)
            ap = relevant/top
            print(query_path)
            print('ap:',ap)
            with open('result/densenet_output.csv', 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([path,ap])
            print(len(feature_list))


            print(image)
            print('ralavent_target : ',ralavent_target)
            from collections import Counter
            element_count = Counter(ralavent_target)
            most_common_element = element_count.most_common(1)
            print("出现最多的元素是：", most_common_element[0][0])
            print("它出现的次数是：", most_common_element[0][1])
            #assert False,print('pass')
            
            #path = path.split('\\')[1]+path.split('\\')[2]+'_'+path.split('\\')[3]
            fig, axs = plt.subplots(3, 4, figsize=(15, 10))
            fig.subplots_adjust(hspace=0.5, wspace=0.3)
            
            axs = axs.flatten()  # 将子图数组展开为一维数组
            plt.suptitle('query most similar : '+ str(most_common_element[0][0]), fontsize=16) # 添加大标题
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
                        a = (image_path.split('*')[0]).split('\\')[-1]
                        axs[i].set_title(f"{i}\n {os.path.basename(image_path.split('*')[1])}\n{a}")
                    f.close()

            # 移除多余的子图
            for i in range(len(image), len(axs)):
                axs[i].remove()
            print(path)
            plt.savefig('result/densenet/'+str( most_common_element[0][1])+'/'+path.split('\\')[-1][:-4]+'.png')
            #plt.show()
            plt.clf()
            plt.close('all')
           # print(pathhhhhhhhhhhhhhhhh)
    # except:
    #      path = 'query_data\S' +"\\"+ file_class + '\\'+filename
    #      print(path)
         #continue#     print(aaa)
#     break
#   break
   
