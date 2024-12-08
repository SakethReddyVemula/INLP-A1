import re

class Tokenizer:
    def __init__(self):
        pass
    
    def tokenize_text(self, text):
        # Sentence Tokenizer
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        tokenized_text = []

        for sentence in sentences:
            tokenized_sentence = self.tokenize_sentence(sentence)
            tokenized_text.append(tokenized_sentence)

        return tokenized_text
    
    def tokenize_sentence(self, sentence):
        # Word Tokenizer
        words = re.findall(r'(?:(?<= )\d+(?:\.\d+)?\%)|(?:[\$]\d+(?:(?:\.\d+)*)?|Rs\.\d+(?:[\.\d+]*))|(?:\d:\d{2} [AP]M)|(?:(?<=[ \n])@[_\w]+)|(?:#[_\w]+)|(?:[\w!%+-\.\/]+@[a-zA-Z0-9\.]*[a-zA-Z0-9]+|".+"@[a-zA-Z0-9\.]*[a-zA-Z0-9]+)|(?:(?:[a-z][a-z0-9+.-]*):\/\/(?:(?:[a-z]+)(?::(?:[^@]+))?@)?(?:[a-zA-Z0-9\.-]+|\[[a-fA-F0-9:]+\])(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?|(?:(?:\w+\.)+(?:\w+))(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?)|(?:\d+.\d+|\d+|-\d+|\+\d+|\.\d+)|(?:[^\w\s])|(?:\w+)', sentence)
        
        # Percentages
        words = [re.sub(r'^\d+(?:\.\d+)?\%$', '<PERC>', word) for word in words]

        # Price
        words = [re.sub(r'^[\$]\d+(?:(?:\.\d+)*)?|Rs\.\d+(?:[\.\d+]*)$', '<PRICE>', word) for word in words]

        # Time
        words = [re.sub(r'^\d:\d{2} [AP]M$', '<TIME>', word) for word in words]

        # Mentions
        words = [re.sub(r'^@[_\w]+$', '<MENTION>', word) for word in words]

        # Hashtags
        words = [re.sub(r'^#[_\w]+$', '<HASHTAG>', word) for word in words]

        # Mail IDs
        words = [re.sub(r'^[\w!%+-\.\/]+@[a-zA-Z0-9\.]*[a-zA-Z0-9]+|".+"@[a-zA-Z0-9\.]*[a-zA-Z0-9]+$', '<MAILID>', word) for word in words]

        # URLs
        words = [re.sub(r'^(?:[a-z][a-z0-9+.-]*):\/\/(?:(?:[a-z]+)(?::(?:[^@]+))?@)?(?:[a-zA-Z0-9\.-]+|\[[a-fA-F0-9:]+\])(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?|(?:(?:\w+\.)+(?:\w+))(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?$', '<URL>', word) for word in words]

        # Numbers
        words = [re.sub(r'^\d+.\d+|\d+|-\d+|\+\d+|\.\d+$', '<NUM>', word) for word in words]

        # Punctuation
        words = [re.sub(r'^[^\w\s\<\>]$', '', word) for word in words]

        return words

def main():
    tokenizer = Tokenizer()
    input_text = input("your text: ")
    tokenized_text = tokenizer.tokenize_text(input_text)
    print("tokenized text: ", tokenized_text)

if __name__ == "__main__":
    main()
