import os
import csv

filename = '../../data/behance_dataset/resnext_feature_list.csv'
fields = ['video_path', 'feature_path']
data_dir1 = '../../data/behance_videos/'
data_dir2 = '../data/behance_videos/'
out_dir = '../data/behance_video_features/resnext101_fps=6/'

with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile)
    # writing the fields 
    csvwriter.writerow(fields)
    video_files = [f for f in os.listdir(data_dir1) if f.endswith('.mp4')]
    # writing the data rows
    for vid in video_files:
        csvwriter.writerow([os.path.join(data_dir2, vid), os.path.join(out_dir, vid[:-4] + '.npy')])