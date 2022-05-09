import os
import json
from tqdm import tqdm
import traceback
from collections import defaultdict
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import multiprocessing

def get_slices(vid_file, timestamps, out_dir):
    timestamps.sort(key=lambda y: y[0])
    try:
        vid_file_name = os.path.split(vid_file)[-1]
        print("slicing for %s timestamps" % len(timestamps))
        for timestamp, windows in timestamps:
            for window in windows:
                start_time = max(0, timestamp-window)
                end_time = timestamp+window
                span = int(window*2)
                out_file = os.path.join(out_dir + '_' + str(span) + 's', vid_file_name[:-4] + ".%s.cut.mp4" % int(timestamp))
                ffmpeg_extract_subclip(vid_file, start_time, end_time, targetname=out_file)

    except Exception as e:
        print(e)
        with open('error_log.txt', 'a+') as f:
            f.write(vid_file + '\n')
            traceback.print_exc(file=f)
            f.write('----------------------------------------------------------' + '\n')


def get_video_slices(video_ids, video2timestamps, video2tmp, out_dir):
    for source in tqdm(video_ids):
        get_slices(video2tmp[source], video2timestamps[source], out_dir)


def splice_videos_for_intent_dataset(out_dir, n_processes=3):

    files = ['./data/intent/both/train.json',
             './data/intent/both/dev.json',
             './data/intent/both/test.json']

    video2tmp = {}
    samples_by_video = defaultdict(lambda: [])

    for f in files:
        if f.endswith('.jsonl'):
            dataset = [json.loads(line.strip()) for line in open(f, 'r').readlines()]
        else:
            dataset = json.load(open(f, 'r'))

        for sample in tqdm(dataset):

            source = sample["session_id"] + '.trans.tsv'
            video_id = sample["video_id"]

            tmp_name = os.path.join(out_dir, video_id.replace("/", "__"))
            if not tmp_name.endswith('.mp4'):
                tmp_name = tmp_name + '.mp4'

            video2tmp[source] = tmp_name
            timestamp = float(sample["timestamp"])
            samples_by_video[source].append([timestamp, [2.5, 5]])

    video_ids = list(video2tmp.keys())
    print("Queued %s videos for splicing" % len(video_ids))
    print("Start %s jobs" % n_processes)

    # sys.exit()

    # jobs = []
    # for i in range(0, n_processes):
    #     process = multiprocessing.Process(target=get_videos,
    #                                       args=(video_ids[int(i*len(video_ids)/n_processes):int((i+1)*len(video_ids)/n_processes)], samples_by_video, out_dir))
    #     jobs.append(process)

    jobs = []
    for i in range(0, n_processes):
        process = multiprocessing.Process(target=get_video_slices,
                                          args=(video_ids[int(i * len(video_ids) / n_processes):int(
                                              (i + 1) * len(video_ids) / n_processes)],
                                                samples_by_video, video2tmp, out_dir))
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

# download_videos_for_intent_dataset('../../data/behance_dataset/', '../../data/behance_videos', n_processes=8)
splice_videos_for_intent_dataset('/nas-hdd/tarbucket/adyasha/datasets/behance', n_processes=4)