def llm_prompts() -> list(str):
    """
    Returns a list of prompts to be passed into an LLM.
    
    Arguments:
        None
        
    Returns:
        list(str): A list of prompts for the LLM as strings.
    """
    prompts = [
         # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
        Sentiment: Negative
        ###
        Tweet: "My day has been 👍"
        Sentiment: Positive
        ###
        Tweet: "This is the link to the article"
        Sentiment: Neutral
        ###
        Tweet: "This new music video was incredibile"
        Sentiment:""",
                """Translate English to French:

        sea otter => loutre de mer

        peppermint => menthe poivrée

        plush girafe => girafe peluche

        cheese =>"""
    ]
    return prompts
