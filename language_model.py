import sys
import re
from collections import Counter
import random
import math
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pickle 

def read_file(file_path):
    file = open(file_path, 'r', encoding='utf-8')
    text = file.read()
    return text

def cleanCorpus(text):
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    # text = re.sub("n\'t", " not", text)
    return text

def splitSentence(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def splitCorpus(text, num_of_test_instances):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    n_sentences = len(sentences)
    # print(n_sentences)
    # num_of_test_instances = int(0.1 * n_sentences)
    test_sentences = random.sample(sentences, num_of_test_instances)
    train_sentences = [sentence for sentence in sentences if sentence not in test_sentences]
    test_text = " ".join(test_sentences)
    train_text = " ".join(train_sentences)
    return train_text, test_text

def tokenize(text):
    # Sentence Tokenizer
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    tokenized_text = []
    for sentence in sentences:
        words = re.findall(r'(?:(?<= )\d+(?:\.\d+)?\%)|(?:[\$]\d+(?:(?:\.\d+)*)?|Rs\.\d+(?:[\.\d+]*))|(?:\d:\d{2} [AP]M)|(?:(?<=[ \n])@[_\w]+)|(?:#[_\w]+)|(?:[\w!%+-\.\/]+@[a-zA-Z0-9\.]*[a-zA-Z0-9]+|".+"@[a-zA-Z0-9\.]*[a-zA-Z0-9]+)|(?:(?:[a-z][a-z0-9+.-]*):\/\/(?:(?:[a-z]+)(?::(?:[^@]+))?@)?(?:[a-zA-Z0-9\.-]+|\[[a-fA-F0-9:]+\])(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?|(?:(?:\w+\.)+(?:\w+))(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?)|(?:\d+.\d+|\d+|-\d+|\+\d+|\.\d+)|(?:[^\w\s])|(?:\w+)', sentence)
        # # Percentages
        words = [re.sub(r'^\d+(?:\.\d+)?\%$', '<PERC>', word) for word in words]
        # # Price
        words = [re.sub(r'^[\$]\d+(?:(?:\.\d+)*)?|Rs\.\d+(?:[\.\d+]*)$', '<PRICE>', word) for word in words]
        # # Time
        words = [re.sub(r'^\d:\d{2} [AP]M$', '<TIME>', word) for word in words]
        # # Mentions
        words = [re.sub(r'^@[_\w]+$', '<MENTION>', word) for word in words]
        # # Hashtags
        words = [re.sub(r'^#[_\w]+$', '<HASHTAG>', word) for word in words]
        # # Mail IDs
        words = [re.sub(r'^[\w!%+-\.\/]+@[a-zA-Z0-9\.]*[a-zA-Z0-9]+|".+"@[a-zA-Z0-9\.]*[a-zA-Z0-9]+$', '<MAILID>', word) for word in words]
        # # URLs
        words = [re.sub(r'^(?:[a-z][a-z0-9+.-]*):\/\/(?:(?:[a-z]+)(?::(?:[^@]+))?@)?(?:[a-zA-Z0-9\.-]+|\[[a-fA-F0-9:]+\])(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?|(?:(?:\w+\.)+(?:\w+))(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?$', '<URL>', word) for word in words]
        # # Numbers
        words = [re.sub(r'^\d+.\d+|\d+|-\d+|\+\d+|\.\d+$', '<NUM>', word) for word in words]
        # # Punctuation
        # words = [re.sub(r'^[^\w\s\<\>]$', '<PUNCT>', word) for word in words]
        tokenized_text.append(words)
    return tokenized_text


def converted_list(original_list):
    converted_list = ['<s>', '<s>']
    for sublist in original_list:
        converted_list.extend(sublist + ['<s>'])
    converted_list.append('<s>')
    return converted_list

def n_grams(n, tokenized_text):
    n_grams_dict = Counter()
    for sentence_tokens in tokenized_text:
        for i in range(len(sentence_tokens) - n + 1):
            n_gram_key = tuple(sentence_tokens[i:i + n])
            n_grams_dict[n_gram_key] += 1
    return n_grams_dict

def countofcounts(ngrams):
    n_Nr = Counter()
    for key, value in ngrams.items():
        n_Nr[value] += 1
    return n_Nr

def trainGT(tokenized_text):
    trigrams = n_grams(3, tokenized_text)
    copy_trigrams = n_grams(3, tokenized_text)
    bigrams = n_grams(2, tokenized_text)
    copy_bigrams = n_grams(2, tokenized_text)
    unigrams = n_grams(1, tokenized_text)

    tri_Nr = countofcounts(trigrams)
    # print(tri_Nr)
    bi_Nr = countofcounts(bigrams)
    uni_Nr = countofcounts(unigrams)
    sorted_tri_Nr = dict(sorted(tri_Nr.items()))
    sorted_bi_Nr = dict(sorted(bi_Nr.items()))
    counter_list = list(sorted_tri_Nr.items())
    counter_list_bi = list(sorted_bi_Nr.items())

    # plt.figure(figsize=(8, 6))
    # plt.plot(list(sorted_tri_Nr.keys()), list(sorted_tri_Nr.values()), marker='o', linestyle='', markersize=5)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('Key')
    # plt.ylabel('tri_Nr value')
    # plt.title('tri_Nr value vs Key (log scale)')
    # plt.grid(True)
    # plt.show()


    Zr_counter = Counter()
    for i, (key, value) in enumerate(counter_list):
        if i == 0:
            prev_key = None
        else:
            prev_key = counter_list[i-1][0]
        if i == len(counter_list) - 1:
            next_key = None
        else:
            next_key = counter_list[i+1][0]

        if prev_key is not None and next_key is not None:
            Zr = value / (0.5 * (next_key - prev_key))
        elif prev_key is None and next_key is not None:
            # print(next_key)
            Zr = value / (0.5 * (next_key - 0))
        elif prev_key is not None and next_key is None:
            Zr = value / (key - prev_key)
        else:
            Zr = value / key
        Zr_counter[key] = Zr

    # print(Zr_counter)

    Zr_counter_bi = Counter()
    for i, (key, value) in enumerate(counter_list_bi):
        if i == 0:
            prev_key = None
        else:
            prev_key = counter_list_bi[i-1][0]
        if i == len(counter_list_bi) - 1:
            next_key = None
        else:
            next_key = counter_list_bi[i+1][0]

        if prev_key is not None and next_key is not None:
            Zr = value / (0.5 * (next_key - prev_key))
        elif prev_key is None and next_key is not None:
            # print(next_key)
            Zr = value / (0.5 * (next_key - 0))
        elif prev_key is not None and next_key is None:
            Zr = value / (key - prev_key)
        else:
            Zr = value / key
        Zr_counter_bi[key] = Zr

    # print(Zr_counter_bi)

    # print("Zr Counter:", Zr_counter)
    # print("Zr_counter")
    # for key, value in Zr_counter.items():
    #     print(f'r: {key}, Nr: {value}')
    # Plotting Zr_counter values vs keys on a log scale
    # plt.figure(figsize=(8, 6))
    # plt.plot(list(Zr_counter.keys()), list(Zr_counter.values()), marker='o', linestyle='', markersize=5)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('Key')
    # plt.ylabel('Zr_counter value')
    # plt.title('Zr_counter value vs Key (log scale)')
    # plt.grid(True)
    # plt.show()

    # Trigram
    firstZeroZr = -1
    for key, value in Zr_counter.items():
        if Zr_counter[key + 1] == 0 and firstZeroZr == -1:
            firstZeroZr = key + 1 
            break

    r_values = np.arange(firstZeroZr - 1, max(Zr_counter.keys()) + 1)
    Zr_values = np.array([Zr_counter.get(r, 0) for r in r_values])
    log_Zr_values = np.log(Zr_values + 1e-10)  # Adding a small offset 1e-10
    log_r_values = np.log(r_values)
    slope, intercept, _, _, _ = linregress(log_r_values, log_Zr_values)
    a = intercept
    b = slope

    # Bigrams
    firstZeroZr_bi = -1
    for key, value in Zr_counter_bi.items():
        if Zr_counter_bi[key + 1] == 0 and firstZeroZr_bi == -1:
            firstZeroZr_bi = key + 1
            break

    r_values_bi = np.arange(firstZeroZr_bi - 1, max(Zr_counter_bi.keys()) + 1)
    Zr_values_bi = np.array([Zr_counter_bi.get(r, 0) for r in r_values_bi])
    log_Zr_values_bi = np.log(Zr_values_bi + 1e-10)  # Adding a small offset 1e-10
    log_r_values_bi = np.log(r_values_bi)
    slope_bi, intercept_bi, _, _, _ = linregress(log_r_values_bi, log_Zr_values_bi)
    a_bi = intercept_bi
    b_bi = slope_bi

    # print("Parameter a:", a)
    # print("Parameter b:", b)
    if(b < -1 and b_bi < -1):
        for key in range(firstZeroZr - 1, max(Zr_counter.keys()) + 2):
            r = key
            Zr_counter[key] = math.exp(a + b * math.log(r))

        for key, value in trigrams.items():
            trigrams[key] = ((value + 1) * Zr_counter[value + 1]) / Zr_counter[value]

        for key in range(firstZeroZr_bi - 1, max(Zr_counter_bi.keys()) + 2):
            r = key
            Zr_counter_bi[key] = math.exp(a_bi + b_bi * math.log(r))

        for key, value in bigrams.items():
            bigrams[key] = ((value + 1) * Zr_counter_bi[value + 1]) / Zr_counter_bi[value]

        probabilities_gt = Counter()

        for key, value in trigrams.items():
            probabilities_gt[key] = value / bigrams[key[:2]]  # Change
            # print(f'trigram: {key}, ProbGT: {probabilities_gt[key]}')

        # N_value = 0
        # for key, value in unigrams.items():
        #     N_value += value
        
        # Nr1_tri = 0
        # for key, value in tri_Nr.items():
        #     if value <= 1.0:
        #         Nr1 += 1

        N_tri = 0
        for key, value in trigrams.items():
            N_tri += 1

        Nr1_tri = 0
        for key, value in tri_Nr.items():
            if value <= 1.0:
                Nr1_tri += 1

        N_bi = 0
        for key, value in bigrams.items():
            N_bi += 1

        Nr1_bi = 0
        for key, value in bi_Nr.items():
            if value <= 1.0:
                Nr1_bi += 1

        N_uni = 0
        for key, value in unigrams.items():
            N_uni += 1
        
        Nr1_uni = 0
        for key, value in uni_Nr.items():
            if value <= 1.0:
                Nr1_uni += 1

        prob_unseen_tri = Nr1_tri / N_tri
        prob_unseen_bi = Nr1_bi / N_bi
        prob_unseen_uni = Nr1_uni / N_bi


        # print(prob_unseen)
        return probabilities_gt, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni, copy_trigrams, copy_bigrams, unigrams

    else:
        return None, None, None, None, None, None, None, None, None, None

def calcLamdas(trigrams, bigrams, unigrams, N):
    l1 = 0
    l2 = 0
    l3 = 0

    for key, value in trigrams.items():
        if value > 0:
            ft1t2t3 = trigrams[key]
            ft1t2 = bigrams[key[:2]]
            ft2t3 = bigrams[key[1:]]
            ft2 = unigrams[key[1:2]]
            ft3 = unigrams[key[2:]]

            # print(key)
            # print(key[:2])
            # print(key[1:])
            # print(key[1:2])
            # print(key[2:])


            if (ft1t2 - 1) != 0:
                val1 = (ft1t2t3 - 1) / (ft1t2 - 1)
            else:
                val1 = 0
            
            if (ft2 - 1) != 0:
                val2 = (ft2t3 - 1) / (ft2 - 1)
            else:
                val2 = 0

            if (N - 1) != 0:
                val3 = (ft3 - 1) / (N - 1)
            else:
                val3 = 0

            if val1 >= val2 and val1 >= val3:
                l3 += ft1t2t3
            elif val2 >= val3 and val2 >= val1:
                l2 += ft1t2t3
            elif val3 >= val1 and val3 >= val2:
                l1 += ft1t2t3
        
    sumli = l1 + l2 + l3
    l1 = l1 / sumli
    l2 = l2 / sumli
    l3 = l3 / sumli
    # l1 = 0.497
    # l2 = 0.435
    # l3 = 0.066
    # print(l1, l2, l3)
    
    return l1, l2, l3


def trainLI(tokenized_text):
    # tokenized_text = handleUNKWords(tokenized_text) # To handle Unknown words
    trigrams = n_grams(3, tokenized_text)
    bigrams = n_grams(2, tokenized_text)
    unigrams = n_grams(1, tokenized_text)
    tri_Nr = countofcounts(trigrams)
    bi_Nr = countofcounts(bigrams)
    uni_Nr = countofcounts(unigrams)
    N_uni = 0
    for key, value in unigrams.items():
        N_uni += value

    N_tri = 0
    for key, value in trigrams.items():
        N_tri += 1

    Nr1_tri = 0
    for key, value in tri_Nr.items():
        if value <= 1.0:
            Nr1_tri += 1

    N_bi = 0
    for key, value in bigrams.items():
        N_bi += 1

    Nr1_bi = 0
    for key, value in bi_Nr.items():
        if value <= 1.0:
            Nr1_bi += 1

    N_uni = 0
    for key, value in unigrams.items():
        N_uni += 1
        
    Nr1_uni = 0
    for key, value in uni_Nr.items():
        if value <= 1.0:
            Nr1_uni += 1

    prob_unseen_tri = Nr1_tri / N_tri
    prob_unseen_bi = Nr1_bi / N_bi
    prob_unseen_uni = Nr1_uni / N_bi
    # print(prob_unseen_tri)
    # print(prob_unseen_bi)
    # print(prob_unseen_uni)

    l1, l2, l3 = calcLamdas(trigrams, bigrams, unigrams, N_uni)
    probabilities_li = Counter()
    for key, value in trigrams.items():
        probabilities_li[key] = l1 * ((unigrams[key[2:]]) / N_uni) + l2 * ((bigrams[key[1:]]) / unigrams[key[1:2]]) + l3 * ((trigrams[key]) / bigrams[key[:2]])
    return probabilities_li, trigrams, bigrams, unigrams, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni

def perplexity(sentence, probabilities, trigrams, bigrams, unigrams, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni):
    unknown_instances = 0
    # print(sentence)
    tokenized_sentence = tokenize(sentence)[0]
    length = len(tokenized_sentence)
    logScore = 0.0
    for i in range(len(tokenized_sentence) - 3 + 1):
        ngram = tuple(tokenized_sentence[i:i + 3])
        prob_ngram = probabilities[ngram]

        if prob_ngram != 0.0:
            logScore += math.log(prob_ngram)
        elif prob_ngram == 0.0:
            # print(f'prob_ngram zero')
            # print(f'unseen ngram: {ngram}')
            unknown_instances += 1
            logScore += math.log(prob_unseen_tri)

    if(len(sentence) >= 2):
        lastbigram = tuple(tokenized_sentence[len(tokenized_sentence) - 2:])
        lastunigram = tuple(tokenized_sentence[len(tokenized_sentence) - 1:])
        if bigrams[lastbigram] != 0.0 and unigrams[lastunigram] != 0.0:
            # logScore += math.log(bigrams[lastbigram] / unigrams[lastunigram])
            logScore += math.log(bigrams[lastbigram]) - math.log(unigrams[lastunigram])
        else:
            # print(f'unseen ngram: {lastbigram}')
            unknown_instances += 1
            logScore += math.log(prob_unseen_bi)
        if N_uni != 0.0 and unigrams[lastunigram] != 0.0:
            # logScore += math.log(unigrams[lastunigram] / N)
            logScore += math.log(unigrams[lastunigram]) - math.log(N_uni)
        else:
            # print(f'unseen ngram: {lastunigram}')
            unknown_instances += 1
            logScore += math.log(prob_unseen_uni)
    
    PP = math.exp(((-1.0)/ (length + 1)) * (logScore))
    return PP, unknown_instances



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Correct Format: python3 linearInterpolation.py i <corpus_path>")
        sys.exit()
    lm_type = str(sys.argv[1])
    corpus_path = sys.argv[2]
    text = read_file(corpus_path)
    text = cleanCorpus(text)
    tokenized_text = tokenize(text)
    num_of_test_instances = 1000
    train_text, test_text = splitCorpus(text, num_of_test_instances)
    tokenized_train_text = tokenize(train_text)
    # tokenized_train_text = converted_list(tokenized_train_text)
    # print(tokenized_train_text)

    if(str(sys.argv[1]) == "g"):
        probabilities_gt, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni, trigrams, bigrams, unigrams =  trainGT(tokenized_train_text)
        # Save model parameters to a file
        model_parameters = {
            'model': "g",
            'corpus': str(corpus_path),
            'probabilities_gt': probabilities_gt,
            'prob_unseen_tri': prob_unseen_tri,
            'prob_unseen_bi': prob_unseen_bi,
            'prob_unseen_uni': prob_unseen_uni,
            'N_tri': N_tri,
            'N_bi': N_bi,
            'N_uni': N_uni,
            'trigrams': trigrams,
            'bigrams': bigrams,
            'unigrams': unigrams,
            'text': text,
            'train_text': train_text,
            'test_text': test_text,
            'tokenized_text': tokenized_text,
            'tokenized_train_text': tokenized_train_text
        }

        with open('language_model.pkl', 'wb') as f:
            pickle.dump(model_parameters, f)

        # print("GT Model parameters saved successfully.")
                

        # print(perplexity("beautiful walk by the side of the water", probabilities_li, trigrams, bigrams, unigrams, N))
        if probabilities_gt != None:
            sumPP = 0.0
            npp = 0.0
            unknown = 0
            for sentence in splitSentence(test_text):
                pp, unknown_instances = perplexity(str(sentence), probabilities_gt, trigrams, bigrams, unigrams, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni)
                # print(f'sentence: {sentence}, pp: {pp}')
                if pp != float('inf'):
                    sumPP += pp
                    npp += 1
                unknown += unknown_instances
            # print("unknown:", unknown)
            # print(sumPP / npp)

            # sumPP = 0.0
            # npp = 0.0
            # unknown = 0
            # for sentence in splitSentence(train_text):
            #     pp, unknown_instances = perplexity(str(sentence), probabilities_gt, trigrams, bigrams, unigrams, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni)
            #     print(f'{sentence}\t{pp}')
            #     if pp != float('inf'):
            #         sumPP += pp
            #         npp += 1
            #     unknown += unknown_instances


            input_sentence = input("input sentence: ")
            input_sentence_pp, _ = perplexity(str(input_sentence), probabilities_gt, trigrams, bigrams, unigrams, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni)
            input_sentence_score = math.exp((-(len(input_sentence) + 1)) * (math.log(input_sentence_pp)))
            print("score:", input_sentence_score)

    elif(str(sys.argv[1]) == "i"):
        # probabilities_gt, trigrams, bigrams, unigrams, size = train(tokenized_train_text)
        probabilities_li, trigrams, bigrams, unigrams, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni =  trainLI(tokenized_train_text)
        # print(type(prob_unseen_tri))
        # Save model parameters to a file
        model_parameters = {
            'model': "i",
            'corpus': str(corpus_path),
            'probabilities_li': probabilities_li,
            'prob_unseen_tri': prob_unseen_tri,
            'prob_unseen_bi': prob_unseen_bi,
            'prob_unseen_uni': prob_unseen_uni,
            'N_tri': N_tri,
            'N_bi': N_bi,
            'N_uni': N_uni,
            'trigrams': trigrams,
            'bigrams': bigrams,
            'unigrams': unigrams,
            'text': text,
            'train_text': train_text,
            'test_text': test_text,
            'tokenized_text': tokenized_text,
            'tokenized_train_text': tokenized_train_text
        }

        with open('language_model.pkl', 'wb') as f:
            pickle.dump(model_parameters, f)

        # print("GT Model parameters saved successfully.")
                
        if probabilities_li != None:
            sumPP = 0.0
            npp = 0.0
            for sentence in splitSentence(test_text):
                pp, _ = perplexity(str(sentence), probabilities_li, trigrams, bigrams, unigrams, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni)
                # print(f'sentence: {sentence}, pp: {pp}')
                if pp != float('inf'):
                    sumPP += pp
                    npp += 1
            # print(sumPP / npp)

            # sumPP = 0.0
            # npp = 0.0
            # for sentence in splitSentence(train_text):
            #     pp, _ = perplexity(str(sentence), probabilities_li, trigrams, bigrams, unigrams, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni)
            #     print(f'{sentence}\t{pp}')
            #     if pp != float('inf'):
            #         sumPP += pp
            #         npp += 1

            input_sentence = input("input sentence: ")
            input_sentence_pp, _ = perplexity(input_sentence, probabilities_li, trigrams, bigrams, unigrams, prob_unseen_tri, prob_unseen_bi, prob_unseen_uni, N_tri, N_bi, N_uni)
            input_sentence_score = math.exp((-(len(input_sentence) + 1)) * (math.log(input_sentence_pp)))
            print("score:", input_sentence_score)

