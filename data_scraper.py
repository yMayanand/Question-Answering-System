import os
import re
import string
import random
import requests
import numpy as np
from glob import glob
from googlesearch import search
from bs4 import BeautifulSoup as BS
from joblib import Parallel, delayed
from urllib.request import urlretrieve

alpha_num = string.ascii_letters + string.digits

user_agents_url = 'https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/User-Agents/UserAgents-IE.txt'

os.makedirs('./tmp', exist_ok=True)
download_path = './tmp/user_agents.txt'

# download user-agents
if not os.path.exists(download_path):
    urlretrieve(user_agents_url, download_path)

# reading user-agents file
with open(download_path, 'r') as f:
    user_agents = f.readlines()
    user_agents = list(map(lambda x: x.strip('\n'), user_agents))

def write(dir, file_name, text):
    """
    this funcition writes paragraph to the file
    """
    path = os.path.join(dir, file_name)
    with open(path, 'w') as f:
        f.write(text)

def return_user_agent():
    """
    this function returns different user agent randomly 
    """
    ua = random.choice(user_agents)
    # using this header to pretend as regular user so that we are not blocked by website
    headers = {
        'User-Agent': ua
    }
    return headers


def extract_para(url, dir, debug=None):
    """
    this function extracts paragraphs from website given its url
    """
    global count
    try:
        f = requests.get(url, headers=return_user_agent(), timeout=3)
    except:
        f'Cannot download {url}'
        return None
    soup = BS(f.content,'lxml')

    paragraphs = soup.find_all('p')
    paragraphs = list(map(lambda x: x.text, paragraphs))

    paragraphs = '\n'.join(paragraphs)

    title = soup.find('title')

    if title:
        if len(title.text) > 100:
            title = None
    
    if title:
        title = title.text.split()
    else:
        title = random.choices(alpha_num, k=6)
        title = "".join(title)

        
    title = '-'.join(title)
    if debug:
        file_name = title + '-' + str(debug) + '.txt'
    else:
        file_name = title + '.txt'

    file_name = file_name.replace('/', '')

    write(dir, file_name, paragraphs)

def split_para(para, group=1):
    sentences = re.split(r'\.[^0-9a-zA-Z]+', para)
    idxs = np.arange(len(sentences))[::group].tolist()
    new_sentences = []
    for i in range(len(idxs)):
        try:
            new_sentences.append('. '.join(sentences[idxs[i]: idxs[i+1]]) + '.')
        except:
            pass
    return new_sentences

def fetch_content(query):
    urls = list(search(query, tld="co.in", num=10, stop=10, pause=2))

    os.makedirs('/content/dump', exist_ok=True)

    for url in tqdm(urls):
        extract_para(url, '/content/dump')
    
    #Parallel(backends='multiprocessing', n_jobs=2)(delayed(extract_para)(url, '/content/dump') for url in tqdm(urls))

    def is_corrupted(file):
        """
        this function check data is extracted properly from website
        """
        with open(file, 'r') as f:
            l = f.readlines()
        if len(l) < 15:
            return True
        else:
            return False

    corrupted_files = []
    for file in glob('./dump/*.txt'):
        if is_corrupted(file):
            corrupted_files.append(file)

    for file in corrupted_files:
        os.remove(file)