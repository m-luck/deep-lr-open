import torch
import heapq
import numpy as np
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

""" 
How to use this:

from transformers_run import GPT2_Adapter
...
gpt_adap = GPT2_Adapter(cuda_avail=True, verbose=True) # All commands run with this class will have the print out for debugging. verbose=False for silence.
input_tensor, tensor_shape = gpt_adap.context_to_nd_prediction_tensor(word_list, k_next=5)
...

"""
class GPT2_Adapter():

    def __init__(self, cuda_avail=False, verbose=False):
        self.cuda_on = cuda_avail
        self.verbose = verbose
        self.model_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_gpt2", "pytorch_model.bin")

    def context_to_flat_prediction_tensor(self, context, k_next=5):
        """
        Returns flattened tensor of k_next one-hotted prediction words and its shape (a tuple).
        """
        return self.context_to_prediction_tensor(context, self.lists_to_flat_tensor, k_next)   

    def context_to_nd_prediction_tensor(self, context, k_next=5):
        """
        Returns flattened tensor of k_next one-hotted prediction words and its shape (a tuple).
        """
        return self.context_to_prediction_tensor(context, lambda x: torch.from_numpy(np.array(x)), k_next)   

    def context_to_prediction_tensor(self, context, shape_manipulator, k_next=5):
        """
        Requires manipulation function passed in to decide shape of tensor.
        Returns tensor of k_next one-hotted prediction words and its shape (a tuple).
        """
        pred = self.get_prediction(context)
        words = self.get_top_k_next_words(k_next, pred)
        one_hotted = [self.word_to_one_hot(word) for word in words]
        tensor = shape_manipulator(one_hotted) # Replace input to this line for more robust tensor information such as nd embeddings. Original is one-hot.
        return tensor, tensor.size()   

    def get_prediction(self, leading_text:str, num_tokens_predict=1):
        """
        Take a set of words, return the latest predicted word and its tokenized index.
        """

        cuda_on = self.cuda_on

        # OPTIONAL: if we want to have more information on what's happening, activate the logger as follows
        import logging
        # logging.basicConfig(level=logging.INFO)

        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        indexed_tokens = tokenizer.encode(leading_text)

        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        model_state_dict = torch.load(self.model_fn)
        # model = GPT2LMHeadModel.from_pretrained('gpt2', state_dict = model_state_dict)
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is important to have reproducible results during evaluation
        model.eval()

        # CUDA options
        if self.cuda_on: 
            tokens_tensor = tokens_tensor.to('cuda')
            model.to('cuda')

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            
        return predictions
        
    # def max_len(): # Get max length of a list of words
        # max_len = 0
        # max_word = None
        # for i, word_token in enumerate(predictions.tolist()[0][0]): 
        #     word = tokenizer.decode(i)
        #     if len(word) > max_len:
        #         max_len = len(word)
        #         max_word = word
        #         print(word)
        # print(max_len, max_word)
        # print(len(predictions.tolist()[0][0]))
        # MAX_LEN:11:'information' * 5 * 28 * 2  

    def word_to_one_hot(self, word):
        # len 27
        chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
                'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', '*']
        
        word = word.strip()
        if len(word) < 11: 
            l_text = list(word)
            for i in range(11 - len(word)):
                l_text.append('*')
            word = ''.join(l_text)
        predicted_vector = [ [1 if chars[i] == c else 0 for i in range(27)] for c in word]
        if self.verbose: 
            [print(i, row) for i, row in enumerate(predicted_vector)]
            print(word)
        return predicted_vector

    def get_top_k_next_words(self, k, predictions):
        val_ind_tuples = [(-pred.item(), i) for i, pred in enumerate(predictions[0,-1,:])]
        heapq.heapify(val_ind_tuples)
        res = []
        # get the predicted next sub-word 
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        for index in range(0,k):
            predicted_index = heapq.heappop(val_ind_tuples)[1]
            # predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
            predicted_word = tokenizer.decode(predicted_index)
            res.append(predicted_word)
        return res

    def lists_to_flat_tensor(self, lists):
        res = torch.from_numpy(np.array(lists))
        return res.view(-1)

if __name__ == "__main__":
    # Will print out flat prediction
    gpt2adap = GPT2_Adapter(cuda_avail=False, verbose=True)
    res, shape = gpt2adap.context_to_flat_prediction_tensor("The world is in a")
    print(res, shape)

