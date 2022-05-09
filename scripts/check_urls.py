import os, json
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import pickle

def read_tool_timeline(file_path):
    tools = []
    tooltimes = []
    tool_lines = [line.strip() for line in open(file_path, 'r', errors='ignore').readlines()]
    for line in tool_lines:
        tokens = line.split('\t')
        tools.append(tokens[2].lower())
        tooltimes.append(float(tokens[1]))
    return tools, tooltimes


def get_mp4(url_link):
    html_text = requests.get(url_link).text
    soup = BeautifulSoup(html_text, 'html.parser')
    for link in soup.find_all('source'):
        if link.attrs['type'] == "video/mp4":
            return(link.attrs['src'])
    return None


def read_transcript(file_path):
    transcript_lines = [line.strip() for line in open(file_path, 'r', errors='ignore').readlines()]
    trans = []
    trans_times = []
    for line in transcript_lines:
        tokens = line.split('\t')
        try:
            # check if the line has the right format i.e. <abs_time> <start> <end> <text>
            trans.append(tokens[-1])
            trans_times.append(round(float(tokens[1]), 2))
        except IndexError:
            continue
    return trans, trans_times


def check_url_behance(data_dir):

    valid_files = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.trans.tsv')]

    print("Loading pickle file")
    d = pickle.load(open('../../../Downloads/belive-2022-01-30.p', 'rb'))
    b = d['belive']
    data_map = {}
    print("%s data files in pickle file" % len(b))
    for v in b:
        data_map[v['video_page_url']] = {'embed_url': v['embed_url'], 'mp4_url': v['stream']['mp4_url']}

    for i, f in tqdm(enumerate(files)):

        # 2. Read the transcript and skip to next file if the file has zero content
        transcript_file = os.path.join(data_dir, f)
        trans, trans_times = read_transcript(transcript_file)

        tool_file = os.path.join(data_dir, f.replace('.trans.tsv', '.tools.tsv'))
        tools, tooltimes = read_tool_timeline(tool_file)

        if not trans or not tools:
            continue

        weblink = open(os.path.join(data_dir, f.replace('.trans.tsv', '.m_url.txt'))).readlines()[0].strip()
        if weblink.startswith('https://livestream-videos-prod'):
            valid_files.append(f)
            continue

        else:
            weblink = open(os.path.join(data_dir, f.replace('.trans.tsv', '.v_url.txt'))).readlines()[0].strip()
            response = requests.get(weblink)
            if response.status_code == 200:
                print('Web site exists')
            else:
                print('%s Web site does not exist' % weblink)
                continue

            try:
                mp4_url = data_map[weblink]['embed_url']
            except KeyError:
                print("Did not find mapping for %s" % weblink)
                continue

            mp4 = get_mp4(mp4_url)
            if mp4 is None:
                print('Not found video for url: %s \n' % weblink)
            else:
                valid_files.append(f)

    with open('./data/behance_files_live.txt', 'w') as f:
        f.write('\n'.join(valid_files))

# check_url_behance('./data/streams-2021-02-03')

def check_if_video_exists():

    with open('./data/intent/both/train_v9.json', 'r') as f:
        dataset = json.load(f)

    samples = []
    for s in dataset:
        video_id = s["video_id"]
        tmp_name = os.path.join('/nas-hdd/tarbucket/adyasha/datasets/behance', video_id.replace("/", "__"))
        if not tmp_name.endswith('.mp4'):
            tmp_name = tmp_name + '.mp4'
        if os.path.exists(tmp_name):
            samples.append(s)

    print("Retained %s samples" % len(samples))
    with open('./data/intent/both/train_v9_vid_available.json', 'w') as f:
        json.dump(samples, f)

check_if_video_exists()