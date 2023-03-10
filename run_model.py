# Import Required Packages
import os
import sys
import fire

from model_setup_utils import setup_model_in_parallel, load_model


# Function to run inference on LLaMA
def run_inference(
    checkpoint_directory: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    maximum_sequence_length: int = 512,
    maximum_batch_size: int = 32
) -> None:
    """
    Generate LLaMA responses using natural language prompts.
    
    Arguments:
        checkpoint_directory (str): Directory to the model checkpoints.
        tokenizer_path (str): Directory to the model tokenizer.
        temperature (float): A parameter that controls the randomness of the outputs. Defaults to 0.8.
        top_p (float): A parameter that controls how many of the highest-probability words are selected to be included in the generated text. Defaults to 0.95.
        maximum_sequence_length (int): Maximum sequence length. Defaults to 512.
        maximum_batch_size (int): Maximum batch size. Defaults to 32.
        
    Returns:
        None
    """
    # Load in local rank and world size, and ensure GPUs are avaialble to perform inference on
    local_rank, world_size = setup_model_in_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    # Load in model and tokenizer using the generator
    generator = load_model(
        checkpoint_directory,
        tokenizer_path,
        local_rank,
        world_size,
        maximum_sequence_length,
        maximum_batch_size
    )
    
    prompts = []
    
    # Retrieve prompt responses from model
    model_results = generator.generate(
        prompts,
        max_gen_len = 512,
        temperature = temperature,
        top_p = top_p
    )
    
    # Print out prompt responses
    for result in model_results:
        print(result)
        print("\n==================================\n")
    

if __name__ == '__main__':
    fire.Fire(run_inference)
