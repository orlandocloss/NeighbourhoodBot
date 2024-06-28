import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese characters
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u200d"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\u3030"
        "\ufe0f"
        "\u2069"
        "\u2066"
        "\u200c"
        "\u25b6"
        "\u23f8"
        "]", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def main(intent, prompt, actions):
    model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # Selecting the appropriate system message based on intent
    if intent == 'problem':
        system_message = "Formally respond in a single, concise paragraph (not bullet points) to users detailing neighbourhood issues shared for local residents. Demonstrate empathy and understanding and offer non-demanding suggestions for resolution through neighborhood initiatives. When appropriate, use specific knowledge or facts to show the resident others may share this problem. Do not ask questions or prompt for responses, do not mention the word count or explicitly acknowledge the format of the response. Do not exceed 80 words and do not use emojis. "

    elif intent == 'information':
        system_message = "Formally respond in a single, concise paragraph (not bullet points) to users querying about local neighbourhood development. Provide clear, helpful information and suggest ways users can engage in their neighborhood, without being prescriptive. When appropriate, speak on the importance of neighbourhood connection. Do not exceed 80 words, do not use emojis and do not ask questions or prompt for responses. Do not mention the word count or explicitly acknowledge the format of the response."

    elif intent == 'action':
        system_message = f"Formally respond in a single paragraph (not bullet points) to users detailing a skill, problem, or action for local neighborhood development. Provide detailed, creative initiatives to the user's prompt by suggesting how they can use each of the following actions: '{actions}' through the 'neighborhood's digital platform' (the user knows what the digital platform is, DO NOT reference other social media). Focus exclusively on the provided actions to facilitate immediate community engagement. Do not use expressive phrases like 'Great!' at the beginning of the response. Do not exceed 80 words, do not ask questions, and do not mention the word count or explicitly acknowledge the format of the response."

    else:
        raise ValueError("Invalid intent specified. Choose 'problem', 'information', or 'action'.")

    prompt_template = f'''[INST] <<SYS>>
    {system_message}
    <</SYS>>
    {prompt}[/INST]
    '''

    print("\n\n*** Generate:")
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(input_ids=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = remove_emojis(response)
    print(response)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate responses based on intent and prompt.')

    # Add arguments for intent and prompt
    parser.add_argument('intent', type=str, help="Specify the intent: 'problem', 'information', or 'action'.")
    parser.add_argument('prompt', type=str, help='The prompt to process.')
    parser.add_argument('actions', type=str, help='String of possible actions', default="")

    # Parse the arguments
    args = parser.parse_args()
    
    # Pass the intent and prompt to the main function
    main(args.intent, args.prompt, args.actions)
