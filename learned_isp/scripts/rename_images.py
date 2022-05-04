import os

val_path_raw = "../raw_images/val/mediatek_raw/"
val_path_rgb = "../raw_images/val/fujifilm/"
for val_path in [val_path_raw, val_path_rgb]:
    for i, img in enumerate(sorted(os.listdir(val_path))):
        rename_cmd = "mv "+val_path+img+" "+val_path+str(i)+".png"
        #os.system(rename_cmd)
        #print(rename_cmd)


paths = sorted(os.listdir(val_path_raw))
for i in range(len(paths)):
    img = str(i)+".png"
    if img not in paths:
        print(img)

