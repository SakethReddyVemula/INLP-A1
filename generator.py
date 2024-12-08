import pickle
import sys
import subprocess
from collections import Counter
import re
import math

def read_file(file_path):
    file = open(file_path, 'r', encoding='utf-8')
    text = file.read()
    return text

def cleanCorpus(text):
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    # text = re.sub("n\'t", " not", text)
    return text

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
        words = [re.sub(r'^[^\w\s\<\>]$', '', word) for word in words]
        tokenized_text.append(words)
    return tokenized_text

def n_grams(n, tokenized_text):
    n_grams_dict = Counter()
    for sentence_tokens in tokenized_text:
        for i in range(len(sentence_tokens) - n + 1):
            n_gram_key = tuple(sentence_tokens[i:i + n])
            n_grams_dict[n_gram_key] += 1
    return n_grams_dict

# def predict_using_ngram(input_sentence, n, tokenized_text, k):
#     tokenized_input = tokenize(input_sentence)[0]
#     if len(input_sentence) < (n - 1):
#         return "Input sentence must have atleast n - 1 words for an n-gram model"

#     ngrams = n_grams(n, tokenized_text)
#     unigrams = n_grams(1, tokenized_text)
#     No_of_ngrams = 0
#     for key, value in ngrams.items():
#         No_of_ngrams += value

#     prefix = tuple(tokenized_input[-(n - 1):])
#     predictions = []

#     sumNgramProb = 0.0

#     for word, _ in unigrams.items():
#         n_gram = prefix + (word[0],)
#         if ngrams[n_gram] != 0.0 and No_of_ngrams != 0.0:
#             sumNgramProb += math.exp(math.log(ngrams[n_gram]) - math.log(No_of_ngrams))
#     for word, _ in unigrams.items():
#         n_gram = prefix + (word[0],)
#         if ngrams[n_gram] != 0.0 and No_of_ngrams != 0.0 and sumNgramProb != 0.0:
#             probability = math.exp(math.log(ngrams[n_gram]) - math.log(No_of_ngrams) - math.log(sumNgramProb))
#         else:
#             probability = 0.0
#         predictions.append((word[0], probability))
    
#     predictions.sort(key=lambda x: x[1], reverse=True)

#     return predictions[:k]

def predict_using_ngram(input_sentence, n, tokenized_text, k):
    
    tokenized_input = tokenize(input_sentence)[0]
    if len(input_sentence) < (n - 1):
        return "Input sentence must have atleast n - 1 words for an n-gram model"

    ngrams = n_grams(n, tokenized_text)

    if n > 1:
        nm1_grams = n_grams(n - 1, tokenized_text)
    unigrams = n_grams(1, tokenized_text)

    No_of_ngrams = 0
    for key, value in ngrams.items():
        No_of_ngrams += value

    if n > 1:
        No_of_nm1grams = 0
        for key, value in nm1_grams.items():
            No_of_nm1grams += value

    prefix = tuple(tokenized_input[-(n - 1):])
    predictions = []

    sumNgramProb = 0.0

    for word, _ in unigrams.items():
        n_gram = prefix + (word[0],)
        if ngrams[n_gram] != 0.0 and No_of_ngrams != 0.0 and n <= 1:
            sumNgramProb += math.exp(math.log(ngrams[n_gram]) - math.log(No_of_ngrams)) #
        elif ngrams[n_gram] != 0.0 and No_of_ngrams != 0.0 and n > 1:
            sumNgramProb += math.exp(math.log(ngrams[n_gram]) - math.log(No_of_ngrams) + math.log(No_of_nm1grams) - math.log(nm1_grams[n_gram[:n - 1]])) 
    for word, _ in unigrams.items():
        n_gram = prefix + (word[0],)
        if ngrams[n_gram] != 0.0 and No_of_ngrams != 0.0 and sumNgramProb != 0.0 and n <= 1:
            probability = math.exp(math.log(ngrams[n_gram]) - math.log(No_of_ngrams) - math.log(sumNgramProb)) #
        elif ngrams[n_gram] != 0.0 and No_of_ngrams != 0.0 and sumNgramProb != 0.0 and n > 1:
            probability = math.exp(math.log(ngrams[n_gram]) - math.log(No_of_ngrams) + math.log(No_of_nm1grams) - math.log(nm1_grams[n_gram[:n - 1]]) - math.log(sumNgramProb)) 
        else:
            probability = 0.0
        predictions.append((word[0], probability))
    
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:k]

def generate_text_using_ngram(length, tokenized_text, k):
    n = int(input("Enter N: "))
    input_sentence = input("Input Sentence: ")
    while length > 0:
        next_word = predict_using_ngram(input_sentence, n, tokenized_text, k)
        for i in range(k):
            if next_word[i][0] not in ['.', '<PERC>', '<PRICE>', '<TIME>', '<MENTION>', '<HASHTAG>', '<MAILID>', '<URL>', '<NUM>']:
                input_sentence += (" " + str(next_word[i][0]))
                break
        length -= 1
    return input_sentence

def predict_using_LM(input_sentence, tokenized_text, k, probabilities_model):
    n = 3
    tokenized_input = tokenize(input_sentence)[0]
    if len(input_sentence) < (n - 1):
        return "Input sentence must have atleast n - 1 words for an n-gram model"

    ngrams = n_grams(n, tokenized_text)
    
    unigrams = n_grams(1, tokenized_text)
    bigrams = n_grams(2, tokenized_text) # What does probabilities_model() return ???????????? Start day

    No_of_ngrams = 0
    for key, value in ngrams.items():
        No_of_ngrams += value

    prefix = tuple(tokenized_input[-(n - 1):])
    predictions = []

    sumNgramProb = 0.0

    for word, _ in unigrams.items():
        n_gram = prefix + (word[0],)
        # print(n_gram)
        if probabilities_model[n_gram] != 0.0:
            sumNgramProb += math.exp(math.log(probabilities_model[n_gram]))
    
    for word, _ in unigrams.items():
        n_gram = prefix + (word[0],)
        if probabilities_model[n_gram] != 0.0 and sumNgramProb != 0.0:
            probability = math.exp(math.log(probabilities_model[n_gram]) - math.log(sumNgramProb))
        else:
            probability = 0.0
        predictions.append((word[0], probability))
    
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:k]

def generate_text_using_LM(input_sentence, length, tokenized_text, k, probabilities_model):
    while length > 0:
        next_word = predict_using_LM(input_sentence, tokenized_text, k, probabilities_model)
        # if str(next_word[0][0]) != ".":
        #     input_sentence += (" " + str(next_word[0][0]))
        # else:
        #     input_sentence += (" " + str(next_word[1][0]))
        # length -= 1
        for i in range(k):
            if next_word[i][0] not in ['.', '<PERC>', '<PRICE>', '<TIME>', '<MENTION>', '<HASHTAG>', '<MAILID>', '<URL>', '<NUM>']:
                input_sentence += (" " + str(next_word[i][0]))
                break
        length -= 1
    return input_sentence

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Correct Format: python3 generator.py <lm_type> <corpus_path> <k>")
        sys.exit()

    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])

    # Form the command to execute
    command = ["python3", "language_model.py", lm_type, corpus_path]

    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command execution failed with return code {e.returncode}")

    with open('language_model.pkl', 'rb') as f:
        model_parameters = pickle.load(f)

    

    input_sentence = input("Input Sentence: ")

    # text = model_parameters['text']
    text = read_file(corpus_path)
    text = cleanCorpus(text)

    # tokenized_text = model_parameters['tokenized_text']
    tokenized_text = tokenize(text)
    # unigrams = model_parameters['unigrams']

    if lm_type == "i":
        probabilities_model = model_parameters['probabilities_li']
    elif lm_type == "g":
        probabilities_model = model_parameters['probabilities_gt']
    

    predictions = predict_using_ngram(input_sentence, 4, tokenized_text, k)
    print(predictions)
        
    # generate_text = generate_text_using_ngram(50, tokenized_text, k)
    # print("output:")
    # print(generate_text)
        
    predictions = predict_using_LM(str(input_sentence), tokenized_text, k, probabilities_model)
    print(predictions)
    for token, prob in predictions:
        print(f'{token}\t{prob}')
    # generate_text = generate_text_using_LM(input_sentence, 50, tokenized_text, k, probabilities_model)
    
    # print(generate_text)
