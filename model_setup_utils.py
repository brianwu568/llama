# Import Required Packages
import os
import sys
import torch
import time
import json
import fire # generate CLIs with one line of code
from typing import Tuple
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel # parallel initialization
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


# Model Setup and Load Functions
def setup_model_in_parallel(seed: int = 69) -> Tuple[int, int]:
    """
    Initialize a model in a parallel fashion. Gets local rank and world size variables from the global config, then initializes the model.
    
    Args:
        seed (int): Random number seed. Defaults to 69.
        
    Returns:
        Tuple[int, int]: A two-element tuple consisting of the local rank and the world size.
    """
    # Get Local Rank and World Size
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    # Initialization
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    
    # Ensure that the seed for the random number generator is the same in every process
    torch.manual_seed(seed)
    result = (local_rank, world_size)
    
    return result


def load_model(
    checkpoint_directory: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    maximum_sequence_length: int,
    maximum_batch_size: int
) -> LLaMA:
    """
    Load a LLaMA model.
    
    Arguments:
        checkpoint_directory (str): Directory to model weights.
        tokenizer_path (str): Directory to the tokenizer.
        local_rank (int): ID of the worker within a node (# of GPUs)
        world_size (int): Maximum number of processes participating in the job.
        maximum_sequence_length (int): Maximum Sequence Length.
        maximum_batch_size (int): Maximum Batch Size.
        
    Returns:
        LLaMA: A generator consisting of a loaded LLaMA model and its associated tokenizer.
    """
    # Load Model Checkpoints
    starting_time = time.time()
    model_checkpoints = sorted(Path(checkpoint_directory).glob('*.pth'))
    
    assert world_size == len(model_checkpoints), f"World Size {world_size} must be equal to the number of checkpoints, which is currently {len(model_checkpoints)}."
    
    checkpoint_path = model_checkpoints[local_rank]

    print("Loading Model from {}".format(checkpoint_directory))
    
    checkpoint = torch.load(checkpoint_path, map_location = "cpu")
    
    # Initialize Model Parameters
    with open(Path(checkpoint_directory) / "params.json", 'r') as parameters_file:
        parameters = json.loads(parameters_file.read())
        
    # Generate Model Arguments
    model_arguments: ModelArgs = ModelArgs(
        max_seq_len = maximum_sequence_length,
        max_batch_size = maximum_batch_size,
        **parameters
    )
    
    # Set up Tokenizer
    tokenizer = Tokenizer(model_path = tokenizer_path)
    model_arguments.vocab_size = tokenizer.n_words
    
    # Set up Model
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_arguments)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict = False)
    
    # Wrap Model and Tokenizer in a Generator, compute loading time, and return Generator
    generator = LLaMA(model, tokenizer)
    current_time = time.time()
    print(f"Current Model was loaded in {current_time - starting_time:.2f} seconds from {checkpoint_directory}.")
    
    return generator
        