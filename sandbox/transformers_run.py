import torch

import heapq
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def predict_n_words(leading_text:str, num_tokens_predict=1):
    """
    Take a set of words, return the latest predicted word and its tokenized index.
    """

    # OPTIONAL: if we want to have more information on what's happening, activate the logger as follows
    import logging
    # logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Encode a text inputs
    text = "The world is"

    for i in range(0,10):
        indexed_tokens = tokenizer.encode(text)

        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])
         
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is important to have reproducible results during evaluation
        model.eval()

        # CUDA options
        # tokens_tensor = tokens_tensor.to('cuda')
        # model.to('cuda')

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        
        print(predictions)

        val_ind_tuples = [(-pred.item(), i) for i, pred in enumerate(predictions[0,-1,:])]

        heapq.heapify(val_ind_tuples)

        # get the predicted next sub-word 
        for index in range(0,num_tokens_predict):
            predicted_index = heapq.heappop(val_ind_tuples)[1]
            predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        text = ' '.join(predicted_text.strip('.,').split(' ')[1:])
        # print(predicted_text)

    # print(predicted_text, "[ Token",predicted_index,"]")
    predicted_word = tokenizer.decode(predicted_index)

    print(''.join(list(predicted_word)))
    return predicted_word, predicted_index

