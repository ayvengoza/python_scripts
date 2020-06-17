from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
from collections import defaultdict
from machine_learning import split_data

from collections import Counter
import re, math, glob, random


def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9']+", text)
    return set(all_words)

class Message(NamedTuple):
    text: str
    is_spam: bool

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
            
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1
    
    def _probabilities(self, token: str) -> Tuple[float, float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)
        
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)

messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

ham_text = "hello any"
spam_text = "hello spam"

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

print(model.predict(ham_text))
print(model.predict(spam_text))

def main():
    path = "resources/spam_data/*/*"

    data: List[Message] = []

    for filename in glob.glob(path):
        is_spam = "ham" not in filename
        with open(filename, errors='ignore') as email_file:
            for line in email_file:
                if line.startswith("Subject: "):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(subject, is_spam))
                    break
    random.seed(0)
    train_messages, test_messages = split_data(data, 0.75)    

    model = NaiveBayesClassifier()
    model.train(train_messages)

    predictions = [(message, model.predict(message.text))
                    for  message in test_messages]
    confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                                for message, spam_probability in predictions)
    print(confusion_matrix)

    def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
        prob_if_spam, prob_if_ham = model._probabilities(token)
        return prob_if_spam / (prob_if_spam + prob_if_ham)

    words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

    print("spamiest_words", words[-10:])
    print("hamiest_words", words[:10])


if __name__ == "__main__":
    main()    
