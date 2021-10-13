import json
import tensorflow as tf
import torch
from math import log

class Tokenizer:
    def __init__(self) -> None:
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 0
        self.name = "オリバー"

        # Taken from https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words
        self.words = open("word_list.txt", encoding="utf-8").read().split()
        self.wordcost = dict((k, log((i+1)*log(len(self.words)))) for i,k in enumerate(self.words))
        self.maxword = max(len(x) for x in self.words)

    # Separate the words
    # Taken from https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words
    def segment(self, s):
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i-self.maxword):i]))
            return min((c + self.wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

        cost = [0]
        for i in range(1,len(s)+1):
            c,k = best_match(i)
            cost.append(c)

        out = []
        i = len(s)
        while i>0:
            c,k = best_match(i)
            assert c == cost[i]
            out.append(s[i-k:i])
            i -= k

        return " ".join(reversed(out))

    # Encode the user's text
    def encode(self, text, return_tensors=None):
        if self.name in text:
            text = text.replace(self.name, "<|name|>")
        output_ids = []
        words = self.segment(text).split()
        for word in words:
            with open('vocab.json', encoding='utf-8') as f:
                vocab = json.load(f)
                if word not in vocab:
                    print("ERROR: Word not found.")
                    print("Issue: " + word)
                    print("Full text: ", end="")
                    print(words)
                    print("Please note this error and send it to coomahsensei@gmail.com")
                    exit()
                for key, value in vocab.items():
                    if key == word:
                        output_ids.append(value)
        if return_tensors == 'tf':
            output = tf.convert_to_tensor(output_ids)
            output = tf.expand_dims(output, axis=0)
        elif return_tensors == 'pt':
            output = torch.IntTensor(output_ids)
            output = torch.unsqueeze(output, 0)
        else:
            output = output_ids
        return output

    # Decode the bot's text
    def decode(self, ids):
        output_text = []
        for id in ids:
            with open('vocab.json', encoding='utf-8') as f:
                vocab = json.load(f)
                for key, value in vocab.items():
                    if value == id:
                        output_text.append(key)
        output = ''.join(output_text)

        if "<|name|>" in output:
            output = output.replace("<|name|>", self.name)
        output = output.split("<|endoftext|>")
        return output[0]