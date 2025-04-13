from importlib.metadata import version
import transformers

from src.llama_model import CustomLlamaAttention, LlamaSdpaAttention
from src.llama_model import prepare_inputs_for_generation_llama_new

def replace_llama(config):
   
    print("Using quick-kv!")
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new
    transformers.models.llama.modeling_llama.LlamaAttention.forward = CustomLlamaAttention.forward
    transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = LlamaSdpaAttention.forward

