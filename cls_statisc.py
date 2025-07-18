from ultralytics import YOLO
import pandas as pd
import cv2

 # 指定图片所在文件夹
img_folder = r"E:\00learning\00test\验证图片"
yolo = YOLO(r"E:\00learning\00test\cls_train\train12\weights\best.pt")
# results = yolo.predict(source=img_folder,stream=True)
results = yolo.predict(source=img_folder,stream=True,max_det=1)
# print(results)
# data= results.to_df()
# df = pd.DataFrame(data)
# print(df)

# # # Process results list
df_out=pd.DataFrame()
for result in results:
    # boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs 
    df = result.to_df()  #df列：Index(['name', 'class', 'confidence', 'box'], dtype='object')    
    df['path'] = result.path
    df_out = pd.concat([df_out,df],ignore_index=True)   
    #图片显示
    # cv2.imshow('img',result.plot())
    # cv2.waitKey(0)
df_out.to_csv('分类检测清单12.csv',index=False)




# # Validate the model
# metrics = yolo.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map  # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps  # a list contains map50-95 of each category



# from ultralytics import settings

# # View all settings
# # print(settings)

# # Return a specific setting
# value = settings["raytune"]
# print(value)
