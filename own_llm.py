import os
import PyPDF2 as pdfreader
import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import math


class Vocab_Generator:
    
    def __init__(self) -> None:
        pass
    
    def text_extractor(self,database_path):
        # Text extraction from PDF files 

        text = ' ' # Empty string to hold all the text to be tokenized 
        for f in os.listdir(database_path):
            with open(os.path.join(database_path,f),'rb') as pdf_file:
                try:
                    pdf_read_obj = pdfreader.PdfReader(pdf_file)
                    
                    for page_num in range(len(pdf_read_obj.pages)):
                        page = pdf_read_obj.pages[page_num]
                        text += page.extract_text()
                        text += "<|endoftext|>" # Add the end of text token 
                except pdfreader.errors.PdfReadError:
                    print(f)
                    continue
        return text

    def vocab_creator(self,text):
        preprocessed_text = re.split(r'([,.?_!"()\']|--|\s)',text)
        preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
        all_tokens = sorted(list(set(preprocessed_text)))
        all_tokens.extend(["<|endoftext|>","<|unk|>"])
        vocab = {token:integer for integer,token in enumerate(all_tokens)}
        return vocab
    

class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()} 

    def encoder(self,text):
        #print("Total number of character:", len(text))
        #print(text[:99])
        preprocessed_text = re.split(r'([,.?_!"()\']|--|\s)',text)
        preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
        preprocessed_text = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed_text]
        #print(len(preprocessed_text))
        ids = [self.str_to_int[s] for s in preprocessed_text]
        return ids

    def decoder(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text

class BytePairEncoding():
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encoder(self,text):
        ids = self.tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        return ids
    
    def decoder(self,ids):
        decoded_text = self.tokenizer.decode(ids)
        return decoded_text 

class DatasetCreator(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            output_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))
    
    # def __len__(self):
    #     return len(self.input_ids)
    
    # def __getitem__(self,idx):
    #     return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader(token_ids, batch_size, max_length, stride, shuffle = True, drop_last=True):
    dataset = DatasetCreator(token_ids, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last)
    return dataloader

def main():
    database_path = "../dataset/"

    # Create the vocabulary after tokenizing the entire text
    vocab_builder = Vocab_Generator()
    raw_text = vocab_builder.text_extractor(database_path)
    vocab = vocab_builder.vocab_creator(raw_text)
    # Convert tokenized text into Token IDs which is later used to create embeddings
    token_gen = "BPE"
    if token_gen == "Simple":
        tokenizer = SimpleTokenizerV1(vocab)
        ids = tokenizer.encoder(raw_text)
    elif token_gen == "BPE":
        tokenizer = BytePairEncoding()
        ids = tokenizer.encoder(raw_text)
    #print(ids)
    # Compute the Hyperparameter values
    max_length = 1024
    overlap_length = 256
    batch_size = math.floor(len(ids)/(2*max_length-overlap_length))
    print(batch_size)
    dataloader = create_dataloader(ids,batch_size,max_length,stride=max_length-overlap_length,shuffle=False)
    #data_iter = iter(dataloader)
    #first_batch = next(data_iter)
    #print(first_batch)
    #print(tokenizer.decoder(ids))



main()