'''
Based on
https://github.com/abetlen/llama-cpp-python

Documentation:
https://abetlen.github.io/llama-cpp-python/
'''

from llama_cpp import Llama, LlamaCache

from modules import shared
from modules.callbacks import Iteratorize


class LlamaCppModel:
    def __init__(self):
        self.initialized = False

    @classmethod
    def from_pretrained(self, path):
        result = self()

        params = {
            'model_path': str(path),
            'n_ctx': 2048,
            'seed': 0,
            'n_threads': shared.args.threads or None,
            'n_batch': shared.args.n_batch
        }
        self.model = Llama(**params)
        self.model.set_cache(LlamaCache)

        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def encode(self, string, return_tensors, add_special_tokens):
        if type(string) is str:
            string = string.encode()
        return self.model.tokenize(string)

    def generate(self, context="", token_count=20, temperature=1, top_p=1, top_k=50, repetition_penalty=1, callback=None):
        if type(context) is str:
            context = context.encode()
        tokens = self.model.tokenize(context)

        output = b""
        count = 0
        for token in self.model.generate(tokens, top_k=top_k, top_p=top_p, temp=temperature, repeat_penalty=repetition_penalty):
            text = self.model.detokenize([token])
            output += text
            if callback:
                callback(text.decode())

            count += 1
            if count >= token_count or (token == self.model.token_eos()):
                break

        return output.decode()

    def generate_with_streaming(self, **kwargs):
        print(kwargs)
        with Iteratorize(self.generate, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply


from modules.ggml.ggmlTextModel import ggmlTextModel
import os

class GPTNeoXCppModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path):
        result = self()

        params = {
            'model_path': str(path),
            'n_threads': shared.args.threads or None,
            'n_batch': shared.args.n_batch,
            'model_type': "stablelm"
        }

        self.model = ggmlTextModel()
        self.model.setting("/".join(os.path.realpath(__file__).split("/")[0:-2])+"/"+params["model_path"], params["n_threads"], params["n_batch"], params["model_type"])

        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result
    
    def encode(self, string):
        return self.model.encode(string)
    
    def generate(self, context="", token_count=20, temperature=1, top_p=1, top_k=50, repetition_penalty=1.2, callback=None):
        output = ""
        print({"context":context, "token_count":token_count, "temperature":temperature, "top_p":top_p, "top_k":top_k, "repetition_penalty":repetition_penalty})
        for text in self.model.generate(n_predict=token_count, top_p=top_p, top_k=top_k, temperature=temperature, seed=-1, repeat_penalty=repetition_penalty, prompt=context):
            if (shared.stop_everything):
                self.model.quit()
            output += text

        return output

    def generate_with_streaming(self, **kwargs):
        reply = ""
        print({"context":context, "token_count":token_count, "temperature":temperature, "top_p":top_p, "top_k":top_k, "repetition_penalty":repetition_penalty})
        for text in self.model.generate(n_predict=kwargs["token_count"], top_p=kwargs["top_p"], top_k=kwargs["top_k"], temperature=kwargs["temperature"], seed=-1, repeat_penalty=kwargs["repetition_penalty"], prompt=kwargs["context"]):
            if (shared.stop_everything):
                self.model.quit()
            reply += text
            yield reply