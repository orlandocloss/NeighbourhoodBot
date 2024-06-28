import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import re
import subprocess
import torch
from torch.cuda.amp import autocast

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
        "\U0001F926-\U0001F937"
        "\U00010000-\U0010FFFF"
        "\u200D"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u23CF"
        "\u23E9"
        "\u231A"
        "\u3030"
        "\uFE0F"
        "\u2069"
        "\u2066"
        "\u200C"
        "\u25B6"
        "\u23F8"
        "]", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def fetch_knowledge_base(prompt):
    try:
        result = subprocess.run(
            ['python3', '/home/orlando/action-scripts/search-disconnection.py', 'search', prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception as e:
        return ""

def main(intent, prompt, actions):
    model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model configuration
    config = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(config, 'quantization_config'):
        config.quantization_config['use_exllama'] = False

    # Load model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        device_map="auto",
        low_cpu_mem_usage=True
    ).to(device)  # Avoid .half() as it's not supported for quantized models

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    knowledge_base = fetch_knowledge_base(prompt)

    # Define the system message based on the intent
    if intent == 'problem':
        system_message = "Formally respond in a single, concise paragraph (not bullet points) to users detailing neighbourhood issues shared for local residents. Demonstrate understanding and offer non-demanding suggestions for resolution through neighborhood initiatives. Do not ask questions or prompt for responses, do not mention the word count or explicitly acknowledge the format of the response. Do not exceed 80 words and do not use emojis."
    elif intent == 'information':
        system_message = "Formally respond in a single, concise paragraph (not bullet points) to users querying about their local neighbourhood. Provide clear, helpful information and suggest ways users can engage in their neighborhood, without being prescriptive. Do not exceed 80 words, do not use emojis and do not ask questions or prompt for responses. Do not mention the word count or explicitly acknowledge the format of the response."
    elif intent == 'action':
        system_message = f"Formally respond in a single paragraph (not bullet points) to users detailing a skill, problem, or action for local neighborhood development. Provide detailed, creative initiatives to the user's prompt by suggesting how they can use each of the following actions: '{actions}' through the 'neighborhood's digital platform' (the user knows what the digital platform is, DO NOT reference other social media). Focus exclusively on the provided actions to facilitate immediate community engagement. Do not use expressive phrases like 'Great!' at the beginning of the response. Do not exceed 80 words, do not ask questions, and do not mention the word count or explicitly acknowledge the format of the response."
    elif intent == 'loneliness':
        system_message = "Formally respond in a single, concise paragraph (not bullet points, 80 words) to users expressing feelings of loneliness. Demonstrate empathy and understanding, without referring to the chatbot as a human. Reference relevant facts or insights about loneliness and community disconnection to provide comfort and perspective. Do not exceed 80 words and do not use emojis. Do not use expressive phrases, do not ask questions, and do not mention the word count or explicitly acknowledge the format of the response."
    elif intent == 'disconnection':
        system_message = "Formally respond in a single, concise paragraph (not bullet points, 80 words) to users asking about disconnection in their neighborhood. Provide informed insights about the causes and effects of community disconnection and suggest ways to foster stronger neighborhood bonds. Use relevant data or facts to enhance the response. Do not exceed 80 words and do not use emojis. Do not use expressive phrases, do not ask questions, and do not mention the word count or explicitly acknowledge the format of the response."
    elif intent == "capability":
        system_message = "Formally respond in a single, concise paragraph (not bullet points, 80 words) to users asking about you. Explain your capabilities and why it is potentially valuable when they prompt for information about you. Do not use expressive phrases, do not ask questions, and do not mention the word count or explicitly acknowledge the format of the response. Here is an explanation of your role: 'You role is the NeighbourhoodBot able to answer informative queries, report problems to the local neighbourhood database and encourage and suggest initiatives for community action utilising the Neighbourhoods Digital Platform these include posting Fundraisers, Petitions, Sharing stories, Starting/joining groups, Starting an event. When users report a problem to the database, you can potentially find someone else to solve these issues linking problems with people that want to take action. You are helpful because you encourage action, suggest solutions and link members of a neighbourhood leading to a more connected community. '"
    else:
        system_message = "Invalid intent specified"

    if intent in ['loneliness', 'disconnection']:
        system_message += f"\n\nRelevant Information:\n{knowledge_base}"

    prompt_template = f'''[INST] <<SYS>>
    {system_message}
    <</SYS>>
    {prompt}[/INST]
    '''

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to(device)
    with autocast():
        output = model.generate(input_ids=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(remove_emojis(response))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate responses based on intent and prompt.')
    parser.add_argument('intent', type=str, help="Specify the intent: 'problem', 'information', 'action', 'loneliness', 'capability' or 'disconnection'.")
    parser.add_argument('prompt', type=str, help='The prompt to process.')
    parser.add_argument('actions', type=str, help='String of possible actions', default="")
    args = parser.parse_args()
    main(args.intent, args.prompt, args.actions)
