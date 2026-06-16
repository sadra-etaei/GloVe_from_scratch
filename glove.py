import numpy as np
from datasets import load_dataset
import re

class GloVe:
    def __init__(self,vocab_size,embed_dim=10,x_max=5,alpha=0.75,learning_rate=0.001):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.alpha=alpha
        self.learning_rate=learning_rate
        self.x_max = x_max


        self.W = (np.random.random((vocab_size,embed_dim))-0.5) / embed_dim
        self.W_tilde = (np.random.random((vocab_size,embed_dim))-0.5) / embed_dim

        self.b = (np.random.random((vocab_size))-0.5) / embed_dim
        self.b_tilde = (np.random.random((vocab_size))-0.5) / embed_dim

        self.grad_W = np.zeros_like(self.W)
        self.grad_W_tilde = np.zeros_like(self.W_tilde)
        self.grad_b = np.zeros_like(self.b)
        self.grad_b_tilde = np.zeros_like(self.b_tilde)

    def f(self,x):
        return (x/self.x_max) ** self.alpha if x < self.x_max else 1
        

    def train(self,co_occurence_matrix,epochs=20,eps=1e-8):
        X = co_occurence_matrix
        ii,jj = X.nonzero()
        X = X[ii,jj]

        print(f"Starting training on {len(X)} non-zero co-occurrence pairs...")

        for epoch in range(epochs):
            total_loss=0
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            

            for idx in indices:
                i,j = ii[idx],jj[idx]

                x = X[idx]

                weight_fn = self.f(x)

                pred = np.dot(self.W[i],self.W_tilde[j]) + self.b[i] + self.b_tilde[j]

                


                diff = pred - np.log(x)


                total_loss += weight_fn *(diff**2)

                self.grad_W[i] += (weight_fn * diff) * self.W_tilde[j]
                self.grad_W_tilde += (weight_fn * diff) * self.W[i]
                self.grad_b[i] += weight_fn*diff
                self.grad_b_tilde[j] += weight_fn*diff

                self.W[i] -= (self.learning_rate / np.sqrt((self.grad_W[i]**2)+eps)) *self.grad_W[i]
                self.W_tilde[j] -= (self.learning_rate / np.sqrt((self.grad_W_tilde[j]**2)+eps)) *self.grad_W_tilde[j]
                self.b[i] -= (self.learning_rate / np.sqrt((self.grad_b[i]**2)+eps)) *self.grad_b[i]
                self.b_tilde[j] -= (self.learning_rate / np.sqrt((self.grad_b_tilde[j]**2)+eps)) *self.grad_b_tilde[j]
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        return self.W  + self.W_tilde

    


def distinct_words(corpus):
    corpus = {word for sentence in corpus for word in sentence}
    vocab = sorted(list(corpus))
    vocab_size = len(vocab)

    return vocab,vocab_size

def build_co_occurence_matrix(corpus,window_size=2):
    vocab,vocab_size = distinct_words(corpus)

    X = np.zeros((vocab_size,vocab_size),dtype=np.float64)

    word2id = {word:i for i,word in enumerate(vocab)}
    id2word = {i:word for i,word in enumerate(vocab)}


    for sentence in corpus:
        tokens = sentence

        for i,target_word in enumerate(tokens):
            target_id = word2id[target_word]
            start = max(0,i-window_size)
            end = min(len(tokens),i+window_size+1)

            for j in range(start,end):
                if i==j :
                    continue
                context_word = tokens[j]
                context_id = word2id[context_word]

                distance = abs(i-j)
                X[target_id,context_id] += 1.0 / distance



                # X[target_id,context_id] += 1

    return X,word2id,vocab_size

imdb_dataset = load_dataset("stanfordnlp/imdb")
wiki_text = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')

START_TOKEN = '<START>'
END_TOKEN = '<END>'

def read_corpus(dataset):

    files = dataset["train"]["text"][:1000]
    return [[START_TOKEN] + [re.sub(r'[^\w]', '', w.lower()) for w in f.split(" ")] + [END_TOKEN] for f in files]

corpus = read_corpus(imdb_dataset)



X,word2id,vocab_size = build_co_occurence_matrix(corpus,2)

print(X,vocab_size)


model = GloVe(vocab_size,50,10,learning_rate=0.005)

weights = model.train(X,10)

























