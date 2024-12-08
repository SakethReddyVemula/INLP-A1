# Implementation Procedures
## Procedure to Execute tokenizer.py
>> python3 tokenizer.py
input sentence: <input_sentence>
output: <tokenized_sentence>

## Procedure to Execute language_model.py for sentence scores
Make sure line 480-491, 538-547 are commented.
And line 494-497, 549-552 are uncommented.
LM1:
>> python3 language_model.py g PrideCorpus.txt
LM2: 
>> python3 language_model.py i PrideCorpus.txt
LM3: 
>> python3 language_model.py g UlyssesCorpus.txt
LM4: 
>> python3 language_model.py i UlyssesCorpus.txt

## Procedure to get average perplexity scores along with sentence and corresponding pp score are printed
Make sure line 480-491, 538-547 are Uncommented.
And line 494-497, 549-552 are Commented.
LM1 Train:
Make sure line 472 is: for sentence in splitSentence(train_text)
line 485: for sentence in splitSentence(train_text)
>> python3 language_model.py g PrideCorpus.txt >> 2022114014_LM1_train-perplexity.txt
LM1 Test:
Make sure line 472 is: for sentence in splitSentence(test_text)
line 485: for sentence in splitSentence(test_text)
>> python3 language_model.py g PrideCorpus.txt >> 2022114014_LM1_test-perplexity.txt
LM2 Train:
Make sure line 532 is: for sentence in splitSentence(train_text)
line 542: for sentence in splitSentence(train_text)
>> python3 language_model.py i PrideCorpus.txt >> 2022114014_LM2_train-perplexity.txt
LM2 Test:
Make sure line 532 is: for sentence in splitSentence(test_text)
line 542: for sentence in splitSentence(test_text)
>> python3 language_model.py i PrideCorpus.txt >> 2022114014_LM2_test-perplexity.txt
LM3 Train:
Make sure line 472 is: for sentence in splitSentence(train_text)
line 485: for sentence in splitSentence(train_text)
>> python3 language_model.py g UlyssesCorpus.txt >> 2022114014_LM3_train-perplexity.txt
LM3 Test:
Make sure line 472 is: for sentence in splitSentence(test_text)
line 485: for sentence in splitSentence(test_text)
>> python3 language_model.py g UlyssesCorpus.txt >> 2022114014_LM3_test-perplexity.txt 
LM4 Train:
Make sure line 532 is: for sentence in splitSentence(train_text)
line 542: for sentence in splitSentence(train_text)
>> python3 language_model.py i UlyssesCorpus.txt >> 2022114014_LM4_train-perplexity.txt
LM4 Test:
Make sure line 532 is: for sentence in splitSentence(test_text)
line 542: for sentence in splitSentence(test_text)
>> python3 language_model.py i UlyssesCorpus.txt >> 2022114014_LM4_test-perplexity.txt

# Notes for generator.py
For all generator runs make sure to model on entire corpus instead of train and test splits.
For this ensure the following in language_model.py:
1. Line 434:     tokenized_train_text = tokenize(text)
2. Make sure line 480-491, 538-547 are commented.	
    And line 494-497, 549-552 are Commented.
## Procedure to predict next words using generator.py
Make sure to Uncomment line 220
Make sure line 472 and 532 in language_model.py: for sentence in splitSentence(test_text)
and make sure line 233, 235, 236 are Uncommented and line line 226, 227, 234, 237, 239 is commented
>> python3 generator.py <lm_type> <corpus_path> <k>
input sentence: <input_sentence>

## Procedure to generate a text of specified length
Make sure to Uncomment line 220
Make sure line 472 and 532 in language_model.py: for sentence in splitSentence(test_text)
and make sure line 247, 249 are Uncommented and rest lines from 246-249 are Commented.
Update line 243 as per your required sentence length: generate_text = generate_text_using_LM(input_sentence, <required_sentence_length>, tokenized_text, k, probabilities_model)
>> python3 generator.py <lm_type> <corpus_path> <k>
input sentence: <input_sentence>

## To generate using N-gram models
Make sure to Comment line 220
Make sure to uncomment lines lines 239-241
and rest from 239-249 are Commented
>> python3 generator.py <random_lm_type> <corpus_path> <random_k>
Enter N: <Enter N for N-gram model>


# Additional Schemas used in Tokenizer
<PERC>  Percentages
<PRICE> Price in Rs. or $
<TIME>  Time in XX:YY A/PM

# Limitations
- I didn't consider '<s>' tags between sentences hence my model, originally, can predict only upto the end of a sentence. After encoutering end of a sentence it starts giving 0.0 probabililty to all unigrams.
- I have temporarily solved this problem, by ignoring '.' as predicted word. I move on through k, until I get a predicted word which does not belong to '.' and any placeholders.
- This temporary fix, although, cannot solve the problem of formation of cycles.




