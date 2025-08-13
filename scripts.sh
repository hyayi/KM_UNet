# dataset=busi
# input_size=256
# python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
# python val.py --name ${dataset}_UKAN 

# dataset=glas
# input_size=512
# python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
# python val.py --name ${dataset}_UKAN 

# dataset=cvc
# input_size=256
# python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
# python val.py --name ${dataset}_UKAN 

dataset=ngtube
input_size=1024
python train.py --arch UKAN --dataset ${dataset} --input_w 768 --input_h 1024 --name ${dataset}_UKAN \
                --image_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3005_NGT_hospitals_resized/imagesTr \
                --mask_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3005_NGT_hospitals_resized/labelsTr \
                --splits_final /data/image/project/ng_tube/nnunet/data/nnUNet_preprocessed/Dataset3005_NGT_hospitals_resized/splits_final.json
# python val.py --name ${dataset}_UKAN 






