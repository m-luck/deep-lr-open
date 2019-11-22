import torch
import heapq
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def get_prediction(leading_text:str, num_tokens_predict=1):
    """
    Take a set of words, return the latest predicted word and its tokenized index.
    """

    # OPTIONAL: if we want to have more information on what's happening, activate the logger as follows
    import logging
    # logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    indexed_tokens = tokenizer.encode(leading_text)

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
        
    return predictions
    
# def max_len():
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

def word_to_one_hot(word):
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
    [print(i, row) for i, row in enumerate(predicted_vector)]
    print(word)
    return predicted_vector

def get_top_k_next_words(k, predictions):
    val_ind_tuples = [(-pred.item(), i) for i, pred in enumerate(predictions[0,-1,:])]
    heapq.heapify(val_ind_tuples)
    res = []
    # get the predicted next sub-word 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    for index in range(0,k):
        predicted_index = heapq.heappop(val_ind_tuples)[1]
        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        predicted_word = tokenizer.decode(predicted_index)
        res.append(predicted_word)
    
    return res

if __name__ == "__main__":
    pred = get_prediction("The world is in a")
    words = get_top_k_next_words(5, pred)
    one_hotted = [word_to_one_hot(word) for word in words]
    print(one_hotted)

