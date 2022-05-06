import os
import cv2

validation_dir = "./data/tdt4265_2022_updated/images/val"
validation_src_dir = "./data/tdt4265_2022_updated/images/val_src"
video_dir = "./raw_videos/"
out_dir = "./data/legol_as_dataset/images/val/"

frames_between_images_in_bundle = 4
images_in_bundle = 4
merged_photo = []

source_image_names = os.listdir(validation_src_dir)

for vid_name in os.listdir(video_dir):
    vid_path = os.path.join(video_dir, vid_name)
    vidcap = cv2.VideoCapture(vid_path)
    success = True
    
    bundle_count = 0

    while success:
        for i in range(frames_between_images_in_bundle*images_in_bundle):
            success, image = vidcap.read()
            if not success: break 
            
            if (i % 4 != 0): continue
            
            crop_factor = int((image.shape[0] - image.shape[1]/2)/2)
            cropped_image = image[crop_factor:-crop_factor, :]
            
            down_width = 256
            down_height = 128
            down_points = (down_width, down_height)
            resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_AREA)
            merged_photo.append(resized_down)
           
        if success and len(source_image_names) > 0: 

            source_image_name = source_image_names.pop(0)

            total_pic = cv2.hconcat([merged_photo[0], merged_photo[1], merged_photo[2], merged_photo[3]])
            image_path = os.path.join(validation_dir, source_image_name)
            cv2.imwrite(image_path, total_pic)

            merged_photo.clear()
            bundle_count += 1