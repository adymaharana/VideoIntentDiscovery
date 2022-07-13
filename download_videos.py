import urllib
import multiprocessing
import progressbar
from bs4 import BeautifulSoup
import requests
import urllib.request
import os, json
import traceback
import sys
from collections import defaultdict
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm
import pickle

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


def get_mp4(url_link):
    html_text = requests.get(url_link).text
    soup = BeautifulSoup(html_text, 'html.parser')
    for link in soup.find_all('source'):
        if link.attrs['type'] == "video/mp4":
            return(link.attrs['src'])
    return None


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


def get_video_slices(video_ids, video2timestamps, out_dir):

    for url in tqdm(video_ids):
#         get_video(url, out_dir="../videos/")
        get_slices(url, video2timestamps[url], out_dir)


def get_videos(video_ids, video2tmp, video2link):

    for video_id in video_ids:
        try:
            tmp_name = video2tmp[video_id]
            if not os.path.exists(tmp_name):
                urllib.request.urlretrieve(video2link[video_id], tmp_name, show_progress)
            else:
                continue
        except Exception as e:
            print(e)
            with open('error_log.txt', 'a+') as f:
                f.write(tmp_name + '\n')
                traceback.print_exc(file=f)
                f.write('----------------------------------------------------------' + '\n')


def download_videos_for_intent_dataset(data_dir, out_dir, n_processes=3):
    # files = ['../data/intent/intent_dataset_train.json',
    #          '../data/intent/intent_dataset_train.json',
    #          '../data/intent/intent_dataset_train.json']

    print("Loading pickle file")
    d = pickle.load(open(os.path.join(data_dir, 'belive-2022-01-30.p'), 'rb'))
    b = d['belive']
    data_map = {}
    for v in b:
        data_map[v['video_page_url']] = {'embed_url': v['embed_url'], 'mp4_url': v['stream']['mp4_url']}

    print([k for k in data_map.keys() if 'Lettering-with-Sydney-Prusso-from-Fresco-Pt2' in k])
    
    # files = [os.path.join(data_dir, 'dataset_v2.jsonl')]
    files = [os.path.join(data_dir, 'train.json')]
    samples_by_video = defaultdict(lambda: [])
    video2tmp = defaultdict(lambda : [])
    video2link = defaultdict(lambda: [])

    behance_count = 0
    livestream_count = 0

    for f in files:
        if f.endswith('.jsonl'):
            dataset = [json.loads(line.strip()) for line in open(f, 'r').readlines()]
        else:
            dataset = json.load(open(f, 'r'))

        for sample in tqdm(dataset):

            try:
                video_id = sample["weblink"]
                source = sample["source"]
            except KeyError:
                source = sample["session_id"] + '.trans.tsv'
                video_id = sample["video_id"]

            if source in video2link:
                continue

            # timestamp = float(sample["timestamp"])
            # print(video_id)
            tmp_name = os.path.join(out_dir, video_id.replace("/", "__"))
            if not tmp_name.endswith('.mp4'):
                tmp_name = tmp_name + '.mp4'
            video2tmp[source] = tmp_name

            if 'behance' in video_id or 'streamprod' in video_id:
                try:
                    mp4_url = data_map[video_id]['embed_url']
                    mp4 = get_mp4(mp4_url)
                    if mp4 is None:
                        with open('error_log.txt', 'a+') as f:
                            f.write('Not found video for url: %s \n' % video_id)
                            f.write('----------------------------------------------------------' + '\n')
                    else:
                        video2link[source] = mp4
                        behance_count += 1
                except Exception as e:
                    pass
            else:
                video2link[source] = video_id
                livestream_count += 1

            # target_names = [tmp_name[:-4] + ".%s.cut.%s.mp4" % (int(timestamp), t) for t in [10, 20]]
            # if all([os.path.exists(fname) for fname in target_names]):
            #     exists += 1
            #     continue
            # else:
            #     samples_by_video[video_id].append((float(timestamp), [5, 10]))

    video_ids = list(video2link.keys())
    print("Queued %s videos for download" % len(video_ids))
    print("Start %s jobs" % n_processes)
    print(behance_count, livestream_count)
    # video_ids = list(samples_by_video.keys())
    # total_videos = sum([len(samples_by_video[k]) for k in video_ids])
    # print("%s clips exist" % exists)
    print("Queued %s clips for splicing" % len(video_ids))

    # sys.exit()

    # jobs = []
    # for i in range(0, n_processes):
    #     process = multiprocessing.Process(target=get_videos,
    #                                       args=(video_ids[int(i*len(video_ids)/n_processes):int((i+1)*len(video_ids)/n_processes)], samples_by_video, out_dir))
    #     jobs.append(process)

    jobs = []
    for i in range(0, n_processes):
        process = multiprocessing.Process(target=get_videos,
                                          args=(video_ids[int(i*len(video_ids)/n_processes):int((i+1)*len(video_ids)/n_processes)], video2tmp, video2link))
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
download_videos_for_intent_dataset('./data/', '/nas-hdd/tarbucket/adyasha/datasets/behance', n_processes=8)