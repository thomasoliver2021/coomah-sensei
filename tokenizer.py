import json
import fugashi
import tensorflow as tf
import torch

class Tokenizer:
    def __init__(self) -> None:
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 2
        self._tagger = fugashi.Tagger()

    def encode(self, text, return_tensors=None):
        output_ids = []
        words = [word.surface for word in self._tagger(text)]
        it = iter(words)
        words = []
        for word in it:
            special_token = ""
            if word == "<":
                special_token += word
                while word != ">":
                    word = next(it)
                    special_token += word
                words.append(special_token)
            else:
                words.append(word)
        for word in words:
            with open('./tokenized_data/vocab.json') as f:
                vocab = json.load(f)
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

    def decode(self, ids):
        output_text = []
        for id in ids:
            with open('./tokenized_data/vocab.json') as f:
                vocab = json.load(f)
                for key, value in vocab.items():
                    if value == id:
                        output_text.append(key)
        output = ''.join(output_text)
        output = output.replace("<pad>", "")
        output = output.split("<|endoftext|>")
        return output[0]