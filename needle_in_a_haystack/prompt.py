# from dotenv import load_dotenv
import os
# import tiktoken
import glob
import json
import yaml
# from anthropic import Anthropic
import numpy as np
import asyncio
from asyncio import Semaphore
from transformers import AutoTokenizer

# load_dotenv()

class Prompter:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 tokenizer,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="/home/syildirim/quick-kv/needle_in_a_haystack/PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 context_lengths_min = 300,
                 context_lengths_max = 200000,
                 context_lengths_num_intervals = 1,
                 context_lengths = None,
                 document_depth_percent_min = 1,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 num_concurrent_requests = 1,
                 final_context_length_buffer = 200,
                 save_dir = "prompts",
                 print_ongoing_status = True):
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.num_concurrent_requests = num_concurrent_requests
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.tokenizer= tokenizer
        self.enc= tokenizer
        self.testing_results = []

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        self.save_dir = save_dir

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    def generate_prompt(self, context, context_length, depth_percent):
        return  f"Read the document and answer this question: {self.retrieval_question}"  +  "\nThe document:\n" + context + '\n'
            
            

    async def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context, context_length, depth_percent)

        context_file_location = f'{self.tokenizer_type}_len_{context_length}_depth_{int(depth_percent*100)}'

        # Save the prompts to file for retesting
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Save the result to file for retesting
        with open(f'{self.save_dir}/{context_file_location}_prompts.json', 'w') as f:
            json.dump(prompt, f)

    # async def generate_context(self, context_length, depth_percent):
    #     # Load up tiktoken so we navigate tokens more easily

    #     # Get your Paul Graham files loaded into a string
    #     context = self.read_context_files()

    #     # Truncate the Paul Graham essays to the context length you desire
    #     context = self.encode_and_trim(context, context_length)

    #     # Insert your random statement according to your depth percent
    #     context = self.insert_needle(context, depth_percent, context_length)

    #     return context
    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        # print("reading files")
        # context = self.read_context_files()
        context = ""
        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        
        return self.enc(text, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids.view(-1).tolist()
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)        

        # # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        # context_length -= self.final_context_length_buffer
        print(f"Context Length: {context_length}")

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            #print(f"depth_percent: {depth_percent}")
            #print(f"len(tokens_context) {len(tokens_context)}")
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            print("insertion point: ", insertion_point)

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]
            print("half tokens: ", self.decode_tokens(tokens_new_context) )

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.encode_text_to_tokens('.')
            period_tokens = [29889]
            #print(f"period_tokens {period_tokens}")
            #print(f"insertion_point {insertion_point}")
            #print(f"Context {tokens_new_context}")
            
            # # Then we iteration backwards until we find the first period
            # while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            #     insertion_point -= 1
            #     tokens_new_context = tokens_context[:insertion_point]
            #print(f"insertion_point aft {insertion_point}")
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        return self.enc(context, truncation=False, return_tensors="pt").input_ids.shape[-1]

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
                break
        return context

    def get_tokens_from_context(self, context):
        return self.enc(context, truncation=False, return_tensors="pt").input_ids.view(-1).tolist()
        
    def decode_tokens(self, tokens, context_length=None):
        decoded = self.enc.decode(tokens[:context_length], skip_special_tokens=True)
        return decoded

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) >= context_length:
            context = self.decode_tokens(tokens, context_length)
            return context
        else:
            print(len(tokens), "is not enough.")
            with open(f"{self.haystack_dir}/philosophy.txt", 'r') as f:
                new_context = f.read()
                # print("fetched more context from file:", f"{self.haystack_dir}/philosophy.txt")
            longer_context = context + new_context 
            return self.encode_and_trim(longer_context, context_length)
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Prompt Generation ...")
        print (f"- Tokenizer: {self.tokenizer_type}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())

def merge_json_to_jsonl(input_folder, output_file):
    with open(output_file, 'w') as jsonl_file:
        for filename in os.listdir(input_folder):
            if filename.endswith('.json'):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    jsonl_file.write(json.dumps(data) + '\n')
        
if __name__ == '__main__':
    with open('config-prompt.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ht = Prompter(
        needle=config['prompt']['needle'],
        haystack_dir=config['prompt']['haystack_dir'],
        retrieval_question=config['prompt']['retrieval_question'],

        context_lengths_min=config['context']['min_len'],
        context_lengths_max=config['context']['max_len'],
        context_lengths_num_intervals=config['context']['interval'],
        context_lengths=config['context']['manually_select_list'],

        document_depth_percent_min=config['document_depth']['min_percent'],
        document_depth_percent_max=config['document_depth']['max_percent'],
        document_depth_percent_intervals=config['document_depth']['interval'],
        document_depth_percents=config['document_depth']['manually_select_list'],
        document_depth_percent_interval_type=config['document_depth']['interval_type'],

        # tokenizer_type=config['tokenizer']['tokenizer_type'],
        # model_name=config['tokenizer']['model_name'],

        save_dir=config['save_dir'],
    )

    ht.start_test()
    input_folder = config['save_dir']
    output_file = input_folder + '.jsonl'
    merge_json_to_jsonl(input_folder, output_file)
