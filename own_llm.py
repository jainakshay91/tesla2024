import os
import PyPDF2 as pdfreader
import re

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
    

class TokenizerV1:
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

def main():
    database_path = "../dataset/"

    # Create the vocabulary after tokenizing the entire text
    vocab_builder = Vocab_Generator()
    raw_text = vocab_builder.text_extractor(database_path)
    vocab = vocab_builder.vocab_creator(raw_text)
    # Convert tokenized text into Token IDs which is later used to create embeddings
    tokenizer = TokenizerV1(vocab)
    ids = tokenizer.encoder(raw_text)
    print(ids)

    print(tokenizer.decoder(ids))



main()