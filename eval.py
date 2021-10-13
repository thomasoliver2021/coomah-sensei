from tokenizer import Tokenizer
from transformers import AutoModelWithLMHead
import itertools
import torch
import time

tokenizer = Tokenizer()
model = AutoModelWithLMHead.from_pretrained("chicaaago/coomaa_sensei")

timestr = time.strftime("%Y%m%d-%H%M%S")
file_name = "./logs/log_" + timestr + ".txt"
file = open(file_name, "w", encoding='utf-8')

print("Hello, welcome to the alpha version of クマー先生!")
print("Enter \"q\" to stop chatting at any time.")

count = 0
for i in range(4):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    user_text = input(">> User: ")
    if user_text == "q":
        print("先生: じゃまた！")
        file.close()
        exit()
    new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')
    file.write("User: " + user_text + "\n")

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if count > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids, max_length=500,
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=100, 
        top_p=0.7,
        temperature=0.8
    ) 
    
    # pretty print last ouput tokens from bot
    bot_text = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0])
    print("先生: {}".format(bot_text))
    file.write("Bot: " + bot_text + "\n")

    count += 1

    if count > 2:
        to_seprate = chat_history_ids.tolist()
        to_seprate = to_seprate[0]

        size = len(to_seprate)
        idx_list = [idx + 1 for idx, val in
            enumerate(to_seprate) if val == 0]
        
        res = [to_seprate[i: j] for i, j in
               zip([0] + idx_list, idx_list + 
               ([size] if idx_list[-1] != size else []))]

        del res[0]
        del res[0]

        final = itertools.chain.from_iterable(res)
        final = list(final)

        final = torch.IntTensor(final)
        chat_history_ids = torch.unsqueeze(final, 0)