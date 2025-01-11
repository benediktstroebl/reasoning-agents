import json
import re
from litellm import completion, acompletion
import argparse
from dotenv import load_dotenv
import weave
import asyncio
from typing import List, Dict
import aiohttp
import aiofiles


load_dotenv()

weave.init("rewrite_user_traces")

async def rewrite_interaction(content: str) -> str:
    system_prompt = """You are an AI assistant helping to edit agent traces and convert <user></user> tags into tool calls. You should use the agent statements to understand the context and what information is being requested. Then either move the question by the agent directly as argument to the ask_user tool or formulate a new question that is clear and specific based on the full context. Do not edit or change any of the other parts of the agent trace that are not related to the <user></user> tags or their immediate context.

    Examples:
    Input: "Welcome to customer service. I can help you track your order. Could you please provide your order ID? <user>12345</user> Thank you. Let me look that up. What was the issue you were having with this order? <user>Items missing</user>"
    Output: "Welcome to customer service. I can help you track your order. <tool>ask_user(question="What is your order ID?") → 12345</tool> Thank you. Let me look that up. <tool>ask_user(question="What issue are you having with this order?") → Items missing</tool>"

    Input: "I'll help you book a flight. When would you like to travel? <user>March 15</user> And what's your destination? <user>London</user> Great choice! How many passengers? <user>2</user>"
    Output: "I'll help you book a flight. <tool>ask_user(question="When would you like to travel?") → March 15</tool> <tool>ask_user(question="What is your destination?") → London</tool> Great choice! <tool>ask_user(question="How many passengers?") → 2</tool>"

    Input: "To process your return, I need some information. Do you have the receipt number? <user>RT-789</user> And what's the reason for return? <user>Wrong size</user> Could you specify which size you received? <user>Medium</user>"
    Output: "To process your return, I need some information. <tool>ask_user(question="What is your receipt number?") → RT-789</tool> <tool>ask_user(question="What is the reason for return?") → Wrong size</tool> <tool>ask_user(question="What size did you receive?") → Medium</tool>"

    Input: "Let me check your order status. <tool>search_orders(customer_id="123") → Order #789 found</tool> I found your order. What seems to be the problem with it? <user>Missing items</user> I'll help investigate that. Which items are missing? <user>The blue shirt</user>"
    Output: "Let me check your order status. <tool>search_orders(customer_id="123") → Order #789 found</tool> I found your order. <tool>ask_user(question="What seems to be the problem with it?") → Missing items</tool> I'll help investigate that. <tool>ask_user(question="Which items are missing?") → The blue shirt</tool>"
    """
    
    response = await acompletion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
    )
    return response.choices[0].message.content.strip()

async def process_task(task: Dict, semaphore: asyncio.Semaphore, file_lock: asyncio.Lock, output_file: str) -> Dict:
    if "output" not in task:
        return task
    
    async with semaphore:
        task["output"] = await rewrite_interaction(task["output"])
        async with file_lock:
            async with aiofiles.open(output_file, 'a') as f:
                await f.write(json.dumps(task) + '\n')
    return task

async def process_file(input_path: str, output_path: str, max_concurrent: int):
    with open(input_path) as f:
        data = json.load(f)
    
    # Clear the output file
    open(output_path, 'w').close()
    
    semaphore = asyncio.Semaphore(max_concurrent)
    file_lock = asyncio.Lock()
    
    tasks = [process_task(task, semaphore, file_lock, output_path) for task in data]
    await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="user_converted.json")
    parser.add_argument("--max-concurrent", type=int, default=30, help="Maximum number of concurrent requests")
    args = parser.parse_args()
    
    asyncio.run(process_file(args.input, args.output, args.max_concurrent))

if __name__ == "__main__":
    main()