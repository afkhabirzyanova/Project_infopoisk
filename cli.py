import time
import re
import argparse

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import fasttext
from rank_bm25 import BM25Okapi

url = 'https://raw.githubusercontent.com/tadgeislamins/poroshki_corpus/main/instance/text_id.csv'
df = pd.read_csv(url, index_col=0)

'''
Предобработка для BM-25 модели. На всякий случай приводим к строчным и удаляем 
пунктуацию, хотя по правилам порошков в них и так всё пишется строчными буквами
без знаков препинания.
'''
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [morph.parse(t)[0].normal_form for t in tokens]
    return tokens

POS_MAP = {
    'NOUN': 'NOUN',
    'ADJF': 'ADJ',
    'ADJS': 'ADJ',
    'COMP': 'ADJ',
    'VERB': 'VERB',
    'INFN': 'VERB',
    'PRTF': 'ADJ',
    'PRTS': 'ADJ',
    'GRND': 'VERB',
    'NUMR': 'NUM',
    'ADVB': 'ADV',
    'NPRO': 'PRON',
    'PRED': 'ADV',
    'PREP': 'ADP',
    'CONJ': 'CCONJ',
    'PRCL': 'PART',
    'INTJ': 'INTJ'
}

'''
Предобработка для word2vec модели. Добавляем после каждого слова его 
частеречный тег.
'''
def preprocess_word2vec(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)

    result = []
    for token in tokens:
        parse = morph.parse(token)[0]
        lemma = parse.normal_form
        pos = parse.tag.POS

        if pos is None:
            continue

        upos = POS_MAP.get(pos)
        if upos is None:
            continue

        key = f'{lemma}_{upos}'
        result.append(key)

    return result

'''
Предобработка для fasttext модели. Не делаем лемматизацию.
'''
def preprocess_fasttext(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

morph = MorphAnalyzer()
# Получаем предобработку для индекса BM-25.
df['preprocessed'] = df['text'].apply(preprocess)
docs = {'ids': list(df['id']), 'docs_original': list(df['text']), 'docs_preprocessed': list(df['preprocessed'])}

# Получаем предобработку для индекса word2vec.
df['preprocessed_word2vec'] = df['text'].apply(preprocess_word2vec)
docs_word2vec = {'ids': list(df['id']), 'docs_original': list(df['text']), 'docs_preprocessed': list(df['preprocessed_word2vec'])}

# Получаем предобработку для индекса fasttext.
df['preprocessed_fasttext'] = df['text'].apply(preprocess_fasttext)
docs_fasttext = {'ids': list(df['id']), 'docs_original': list(df['text']), 'docs_preprocessed': list(df['preprocessed_fasttext'])}

# Функция для построения индекса BM-25.
def build_bm25_index(docs):
    return {
        'idx_type': 'bm25',
        'doc_ids': docs['ids'],
        'docs_original': docs['docs_original'],
        'bm25': BM25Okapi(docs['docs_preprocessed'])
    }

# Функция для построения индекса word2vec.
def text_to_vector(tokens, model):
    vectors = []

    for token in tokens:
        if token in model.key_to_index:
            vectors.append(model[token])

    if not vectors:
        return None

    return np.mean(vectors, axis=0)

def build_word2vec_index(docs, model):
    docs_tokens = docs['docs_preprocessed']

    doc_vectors = []
    for tokens in docs_tokens:
        vec = text_to_vector(tokens, model)
        doc_vectors.append(vec)

    return {
        'idx_type': 'word2vec',
        'doc_ids': docs['ids'],
        'docs_original': docs['docs_original'],
        'doc_vectors': doc_vectors,
        'model': model
    }

# Функция для построения индекса fasttext.
def text_to_vector_fasttext(tokens, model):
    vectors = []

    for token in tokens:
        vectors.append(model.get_word_vector(token))

    if not vectors:
        return None

    return np.mean(vectors, axis=0)

def build_fasttext_index(docs, model):
    docs_tokens = docs['docs_preprocessed']

    doc_vectors = []
    for tokens in docs_tokens:
        vec = text_to_vector_fasttext(tokens, model)
        doc_vectors.append(vec)

    return {
        'idx_type': 'fasttext',
        'doc_ids': docs['ids'],
        'docs_original': docs['docs_original'],
        'docs_preprocessed': docs_tokens,
        'doc_vectors': doc_vectors,
        'model': model
    }

def search(query, index, top_k=5):
    doc_ids = index['doc_ids']
    docs_original = index['docs_original']

    if index['idx_type'] == 'bm25':
        scores = np.array(index['bm25'].get_scores(preprocess(query)))

    elif index['idx_type'] == 'word2vec':
        query_vec = text_to_vector(preprocess_word2vec(query), index['model'])

        if query_vec is None:
            end_time = time.perf_counter()
            return [], end_time - start_time

        scores = np.full(len(index['doc_vectors']), -1.0)

        for i, doc_vec in enumerate(index['doc_vectors']):
            if doc_vec is not None:
                scores[i] = cosine_similarity(query_vec.reshape(1, -1), doc_vec.reshape(1, -1))[0][0]

    elif index['idx_type'] == 'fasttext':
        query_tokens = preprocess_fasttext(query)
        query_vec = text_to_vector_fasttext(query_tokens, index['model'])

        if query_vec is None:
            end_time = time.perf_counter()
            return [], end_time - start_time

        scores = np.full(len(index['doc_vectors']), -1.0)

        for i, doc_vec in enumerate(index['doc_vectors']):
            if doc_vec is not None:
                score = cosine_similarity(query_vec.reshape(1, -1), doc_vec.reshape(1, -1))[0][0]

                # опциональное улучшение для лучшего учёта полных совпадений слов
                doc_tokens = index['docs_preprocessed'][i]
                overlap = len(set(query_tokens) & set(doc_tokens))
                if overlap > 0:
                    score += 0.1 * overlap

                scores[i] = score
                
    top_ids = scores.argsort()[::-1][:top_k]

    res = []
    for i in top_ids:
        res.append((int(doc_ids[i]), float(scores[i]), docs_original[i]))
    return res

def get_index(index_name):
    if index_name == 'bm25':
        return build_bm25_index(docs)

    elif index_name == 'word2vec':
        model_word2vec = api.load('word2vec-ruscorpora-300')
        return build_word2vec_index(docs_word2vec, model_word2vec)

    elif index_name == 'fasttext':
        model_fasttext = fasttext.load_model('cc.ru.300.bin')
        return build_fasttext_index(docs_fasttext, model_fasttext)

def run_search(query, index_name, top_k=5):
    print('=' * 40)
    print(f'Query: {query}')
    print(f'Index: {index_name}')
    print('=' * 40)
    
    start_time = time.perf_counter()
    
    index_obj = get_index(index_name)
    results = search(query, index_obj, top_k=top_k)

    end_time = time.perf_counter()
    search_time = end_time - start_time
    
    print(f'\nSearch time: {search_time:.6f} sec')

    if not results:
        print('No results found')
    else:
        print('\n=== RESULTS ===')
        for i, (doc_id, score, text) in enumerate(results, 1):
            print(f'{i}. [id={doc_id}] score={score:.4f}')
            print(text.strip())
            print()

def run_cli():
    parser = argparse.ArgumentParser(description='Поиск по корпусу порошков')

    parser.add_argument('--query', type=str, required=True, help='Текст запроса')
    parser.add_argument('--index', type=str, required=True, choices=['bm25', 'word2vec', 'fasttext'], help='Тип индекса')
    parser.add_argument('--top_k', type=int, default=5, help='Количество результатов')

    args = parser.parse_args()

    run_search(args.query, args.index, args.top_k)


if __name__ == '__main__':
    run_cli()
