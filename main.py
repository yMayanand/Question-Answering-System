import os
import re
from glob import glob
import torch
from data_scraper import split_para, fetch_content
import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

sentence_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4")

def get_similar_content(query):

    content = []
    docs = glob('./dump/*.txt')

    for doc in docs:
        with open(doc, 'r') as f:
            tmp = f.read()
            tmp = re.sub(r' +', ' ', tmp).strip()
            tmp = re.sub(r'\n+', ' ', tmp).strip()
            tmp = re.sub(r'-', ' ', tmp)
            tmp = split_para(tmp, group=5)
            content.extend(tmp)

    # similarity between query and sentences
    #Compute embedding for both lists
    embeddings1 = sentence_model.encode(query, convert_to_tensor=True)
    embeddings2 = sentence_model.encode(content, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    # take top 3 docs
    try:
        idxs = torch.topk(cosine_scores[0], k=3).indices.tolist()
    except:
        print('trying top1')
        idxs = torch.topk(cosine_scores[0], k=1).indices.tolist()

    similar_sent = [content[i] for i in idxs]
    #similar_sent = '\n\n'.join(similar_sent)
    print(similar_sent)

    delete_docs = glob('./dump/*')
    for i in delete_docs:
        os.remove(i)

    return similar_sent

def create_prompt(query, similar_sent):
    prompt = f"""Generate Answer to the Question from the Supporting Texts.
Supporting Texts:-
Supporting Text 1:- NLP Cloud developed their API by mid-2020 and they added many pre-trained open-source models since then.
Supporting Text 2:- NLP Cloud is an Artificial Intelligence Company.
Question:- What type of Company is NLP Cloud?
Answer:- NLP is an  Artificial Intelligence Company.
###
Supporting Texts:-
Supporting Text 1:- NLP Cloud developed their API by mid-2020 and they added many pre-trained open-source models since then.
Supporting Text 2:- NLP Cloud is an Artificial Intelligence Company.
Supporting Text 3:- All plans can be stopped anytime. You only pay for the time you used the service. In case of a downgrade, you will get a discount on your next invoice.
Question:- When can plans be stopped?
Answer:- Plans can be stopped anytime.
###
Supporting Texts:-
Supporting Text 1:- The main challenge with GPT-J is memory consumption. Using a GPU plan is recommended.
Supporting Text 2:- Bill Gates is the Richest person in the world.
Supporting Text 3:- NLP Cloud is an Artificial Intelligence Company.
Question:- Who is the richest person in the world?
Answer:- Bill Gates is the richest person in the world.
###
Supporting Texts:- 
Supporting Text 1:-Apple stock price predictions for September 2024. The forecast for beginning of September 286. Maximum value 310, while minimum 274. Averaged Apple stock price for month 291. Price at the end 292, change for September 2.10%.
Supporting Text 2:-Apple stock price predictions for September 2022. The forecast for beginning of September 158. Maximum value 164, while minimum 140. Averaged Apple stock price for month 153. Price at the end 149, change for September  5.70%.
Supporting Text 3:-Apple stock predictions for October 2022. The forecast for beginning of October 149. Maximum value 155, while minimum 137. Averaged Apple stock price for month 147. Price at the end 146, change for October  2.01%.
Question:-What is apple stock price for september 2022?
Answer:- Apple's stock price for september 2022 is 158.
###
Supporting Texts:-
{similar_sent}
Question:- {query}?
Answer:-"""
    return prompt

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0)

def ask(query=''):
    global generator
    fetch_content(query)

    similar_sent = get_similar_content(query)
    

    tmp_store = []
    for i, j in enumerate(similar_sent):
        tmp = f"Supporting Text {i + 1}:- "
        tmp += j
        tmp_store.append(tmp)

    similar_sent = '\n'.join(tmp_store)

    prompt = create_prompt(query, similar_sent)

    answer = generator(prompt, do_sample=False, min_length=50, max_new_tokens=200,  temperature=0.7, eos_token_id=21017)
    print(answer[0]['generated_text'])
    return answer[0]['generated_text'].split(query)[1].split('###')[0].split('Question:-')[0].split('Answer:-')[1].split('\n')[0].strip()

if __name__ == '__main__':
    demo = gr.Interface(
    fn=ask, 
    inputs="text", 
    outputs=[gr.Textbox(label='Answer')],
    title='Question Answering System',
    description='Ask your Question'
    )

    demo.launch()
