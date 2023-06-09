'''
Edited by Zuo Huichen, Zhou Lei 2023/4/30 21:57
'''

from gensim import corpora
from prettytable import PrettyTable
from collections import Counter
from gensim.models.ldamodel import LdaModel
import pandas as pd

num_topics = 9
patience = 100

PATH = "./dataset/dataset-CalheirosMoroRita-2017.csv"
data_set = []
data = pd.read_csv(PATH, header=0, encoding='windows-1252')
hospitality_dictionary = {'travel': 'tourism', 'tour': 'tourism', 'trip': 'tourism', 'accommodation': 'hospitality',
                          'hotel': 'hospitality', 'resort': 'hospitality', 'lodge': 'hospitality',
                          'decorative': 'decoration', 'interior design': 'decoration', 'architecture': 'decoration',
                          'nature': 'environment', 'sustainability': 'environment', 'vacation': 'holiday',
                          'restaurant': 'food', 'taste': 'food', 'flavors': 'food', 'wine': 'food', 'cuisine': 'food',
                          'meal': 'food', 'employees': 'people', 'staff': 'people', 'workers': 'people',
                          'place': 'location', 'sight': 'location', 'scenery': 'location', 'clients': 'guests',
                          'hosts': 'guests', 'costumers': 'guests', 'website': 'site', 'browsing': 'site',
                          'internet': 'site', 'social networks': 'site', 'relaxed': 'relax', 'calm': 'relax',
                          'quiet': 'relax', 'chill': 'relax', 'sensations': 'feelings', 'sense': 'feelings',
                          'euros': 'eur', 'money': 'eur', 'expensive': 'eur', 'cost': 'eur', 'price': 'eur',
                          'reservation': 'reserve', 'booking': 'reserve', 'availability': 'reserve',
                          'kindness': 'friendly',
                          'caring': 'friendly', 'attentive': 'friendly', 'empathy': 'friendly', 'sympathy': 'friendly',
                          'pleasant': 'friendly', 'creativity': 'different', 'unique': 'different',
                          'singular': 'different',
                          'innovator': 'different', 'original': 'different', 'love': 'romance', 'romantic': 'romance',
                          'passion': 'romance', 'amenities': 'equipment', 'facilities': 'equipment', 'theme': 'trends',
                          'chic': 'trends',
                          'exotic': 'trends', 'hippie': 'trends', 'style': 'trends', 'lounge': 'trends'}
sentiment_dictionary = {'brilliant': 'Strong Positive', 'excellent': 'Strong Positive', 'fantastic': 'Strong Positive',
                        'phenomenal': 'Strong Positive', 'wonderful': 'Strong Positive', 'superb': 'Strong Positive',
                        'beautiful': 'Strong Positive',
                        'spectacular': 'Strong Positive', 'delightful': 'Strong Positive',
                        'memorable': 'Strong Positive', 'remarkably': 'Strong Positive', 'stunning': 'Strong Positive',
                        'cool': 'Ordinary Positive',
                        'good': 'Ordinary Positive', 'fashionable': 'Ordinary Positive', 'helpful': 'Ordinary Positive',
                        'peaceful': 'Ordinary Positive', 'beauty': 'Ordinary Positive', 'quality': 'Ordinary Positive',
                        'warm': 'Ordinary Positive', 'respect': 'Ordinary Positive', 'tasty': 'Ordinary Positive',
                        'recommend': 'Ordinary Positive', 'spacious': 'Ordinary Positive',
                        'pleasure': 'Ordinary Positive', 'elegant': 'Ordinary Positive', 'sincere': 'Ordinary Positive',
                        'liked': 'Ordinary Positive', 'bad': 'Ordinary Negative', 'nervous': 'Ordinary Negative',
                        'loss': 'Ordinary Negative', 'aversion': 'Ordinary Negative', 'sad': 'Ordinary Negative',
                        'difficulty': 'Ordinary Negative', 'quite small': 'Ordinary Negative',
                        'little scattered': 'Ordinary Negative', 'expensive': 'Ordinary Negative',
                        'shame': 'Ordinary Negative',
                        'unbalanced': 'Ordinary Negative', 'spoiled': 'Ordinary Negative',
                        'apology': 'Ordinary Negative', 'terrible': 'Strong Negative', 'awful': 'Strong Negative',
                        'stupid': 'Strong Negative', 'horrible': 'Strong Negative', 'unfortunately': 'Strong Negative',
                        'ridiculous': 'Strong Negative',
                        'really hard': 'Strong Negative', 'too long': 'Strong Negative',
                        'weaknesses': 'Strong Negative', 'very bad': 'Strong Negative'}
datas = []
column_names = list(data.columns)[0:7]
content = data.values
for i in range(content.shape[0]):
    tmp_str = ''
    if i < len(column_names):
        tmp_str = tmp_str + column_names[i] + ': '
    for j in range(content.shape[1]):
        if isinstance(content[i][j], str):
            tmp_str = tmp_str + content[i][j] + ' '
    seg_list = tmp_str.split()
    datas.append(seg_list)
data_set = []
for review in datas:
    t = []
    for word in review:
        l = len(word)
        for key in hospitality_dictionary.keys():

            if len(key) <= l:#
                if key == word[0:len(key)]:
                    if key not in t:
                        t.append(key)
        for key in sentiment_dictionary.keys():
            if len(key) <= l:#
                if key == word[0:len(key)]:
                    if key not in t:
                        t.append(key)
        for val in hospitality_dictionary.values():
            if len(val) <= l:#
                if val == word[0:len(val)]:
                    if val not in t:
                        t.append(val)
        for val in sentiment_dictionary.values():
            if len(val) <= l:  #
                if val == word[0:len(val)]:
                    if val not in t:
                        t.append(val)
    data_set.append(t)
print(data_set)

# Building dictionary
dictionary = corpora.Dictionary(data_set)
corpus = [dictionary.doc2bow(text) for text in data_set]  # Represents the number of times the word appears

# Building the LDA Model
ldaModel = LdaModel(corpus, num_topics=9, id2word=dictionary, passes=50, random_state=2)  # is divided into 9 topics

# Count the number of texts corresponding to different topics Edited by Zuo huichen, Zhou Lei 2023/4/30 21:57
topic_frequancy = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
for i in range(len(corpus)):
    topics_probs = ldaModel.get_document_topics(corpus[i])
    idx = max(topics_probs, key=lambda x: x[1])[0]
    topic_frequancy[idx] += 1


print('print topics!')
result = {}
beta_ = ldaModel.expElogbeta

for i in range(num_topics):
    flag = -1
    res = ldaModel.get_topic_terms(i, patience)
    res.reverse()
    hos_term = None
    hos_word = None
    senti_term = None
    senti_word = None
    for p in res:
        if flag > 0:
            break
        idx, value = p
        word = dictionary.id2token[idx]
        if word in hospitality_dictionary.keys() and hos_term is None:
            hos_term = hospitality_dictionary[word]
            hos_word = word
            flag += 1
        elif word in sentiment_dictionary.keys() and senti_term is None:
            senti_term = sentiment_dictionary[word]
            senti_word = word
            flag += 1
        else:
            continue
    if senti_word is None:
        senti_word = hos_word
    result[i] = {'#': topic_frequancy[i], 'hospitality term': hos_term,
                 'beta_1': beta_[i][dictionary.token2id[hos_word]] * 100, 'sentiment term': senti_term,
                 'beta_2': beta_[i][dictionary.token2id[senti_word]] * 100}
print(result)
table = PrettyTable()
table.field_names = ["#", "Hospitality_term", "beta_1", "sentiment term", "beta_2"]

for val in result.values():
    row = []
    for v in val.values():
        row.append(v)
    table.add_row(row)
print(table)
# ----------------------------------------------The above model for the theme-------------------------------------------

data_set2 = []
for i in range(content.shape[0]):
    for j in range(len(content[i])):
        data_set2.append(content[i][j])

new_data_set = [a_ for a_ in data_set2 if a_ == a_]

text = ". ".join(new_data_set)
filtered_data_set = []
for key in sentiment_dictionary.keys():
    t = []
    count = text.lower().count(key)
    for i in range(count):
        t.append(sentiment_dictionary[key])
    filtered_data_set.append(t)

valset = []
for val in sentiment_dictionary.values():
    if val not in valset:
        valset.append(val)
        t = []
        count = text.lower().count(val)
        for i in range(count):
            t.append(val)
        filtered_data_set.append(t)

for key in hospitality_dictionary.keys():
    t = []
    count = text.lower().count(key)
    for i in range(count):
        t.append(hospitality_dictionary[key])
    filtered_data_set.append(t)
for val in hospitality_dictionary.values():
    if val not in valset:
        valset.append(val)
        t = []
        count = text.lower().count(val)
        for i in range(count):
            t.append(val)
        filtered_data_set.append(t)


word_counts = Counter()
for text in filtered_data_set:
    word_counts.update(text)

# Create a DataFrame with term frequency
term_frequency = pd.DataFrame.from_dict(word_counts, orient='index', columns=['frequency'])
term_frequency.index.name = 'term'
term_frequency.reset_index(inplace=True)

# Sort by frequency (descending)
term_frequency = term_frequency.sort_values(by='frequency', ascending=False)

# Reset index
term_frequency.reset_index(drop=True, inplace=True)

print(term_frequency)