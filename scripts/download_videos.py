import urllib
import multiprocessing
import progressbar
import urllib.request
import os, json
import traceback
import sys
from collections import defaultdict
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = '0'


pbar = None
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def get_slices(url, timestamps, out_dir):
    timestamps.sort(key=lambda y: y[0])
    tmp_name = os.path.join(out_dir, url.replace("/", "__"))
    print("Downloading %s" % url)
    try:
        urllib.request.urlretrieve(url, tmp_name, show_progress)
        print("slicing for %s timestamps" % len(timestamps))
        for timestamp, windows in timestamps:
            for window in windows:
                start_time = max(0, timestamp-window)
                end_time = timestamp+window
                span = window*2
                ffmpeg_extract_subclip(tmp_name, start_time, end_time, targetname=tmp_name[:-4] + ".%s.cut.%s.mp4" % (int(timestamp), span))
        os.remove(tmp_name)
    except Exception as e:
        print(e)
        with open('error_log.txt', 'a+') as f:
            f.write(url + '\n')
            traceback.print_exc(file=f)
            f.write('----------------------------------------------------------' + '\n')

def get_videos(video_ids, video2timestamps, out_dir):

    for url in tqdm(video_ids):
#         get_video(url, out_dir="../videos/")
        get_slices(url, video2timestamps[url], out_dir)


def download_videos_for_intent_dataset(data_dir, out_dir, n_processes=3):
    # files = ['../data/intent/intent_dataset_train.json',
    #          '../data/intent/intent_dataset_train.json',
    #          '../data/intent/intent_dataset_train.json']
    
    files = [os.path.join(data_dir, 'train.json'),
             os.path.join(data_dir, 'dev.json'),
             os.path.join(data_dir, 'test.json')]
    samples_by_video = defaultdict(lambda: [])
    exists = 0
    for f in files:
        dataset = json.load(open(f, 'r'))
        for sample in dataset:
            video_id = sample["video_id"]
            assert video_id.endswith('.mp4'), video_id
            timestamp = float(sample["timestamp"])
            # print(video_id)
            tmp_name = os.path.join(out_dir, video_id.replace("/", "__"))
            assert tmp_name.endswith('.mp4'), tmp_name
            target_names = [tmp_name[:-4] + ".%s.cut.%s.mp4" % (int(timestamp), t) for t in [10, 20]]
            if all([os.path.exists(fname) for fname in target_names]):
                exists += 1
                continue
            else:
                samples_by_video[video_id].append((float(timestamp), [5, 10]))
    
    print("Queued %s videos for download" % len(samples_by_video))
    print("Start %s jobs" % n_processes)
    video_ids = list(samples_by_video.keys())
    total_videos = sum([len(samples_by_video[k]) for k in video_ids])
    print("%s clips exist" % exists)
    print("Queued %s clips for splicing" % total_videos)

    jobs = []
    for i in range(0, n_processes):
        process = multiprocessing.Process(target=get_videos,
                                          args=(video_ids[int(i*len(video_ids)/n_processes):int((i+1)*len(video_ids)/n_processes)], samples_by_video, out_dir))
        jobs.append(process)

    # Start the processes (i.e. calculate the random number lists)
    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for j in jobs:
        j.join()


# get_vids(
#     "http://streamprod-eastus-streamprodeastus-usea.streaming.media.azure.net/8efc8a51-0329-4272-abc7-b12630d7dbd1/output.mp4",
#     10, 20)

download_videos_for_intent_dataset('../../data/behance_dataset/', '../../data/behance_videos', n_processes=8)