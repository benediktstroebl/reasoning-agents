import json
import argparse
from litellm import completion, acompletion
from dotenv import load_dotenv
import weave
import asyncio
import aiofiles
from typing import Dict, List

load_dotenv()
weave.init("add_cot_reasoning")

async def add_cot_reasoning(content):
    system_prompt = """You are an AI assistant helping to add Chain-of-Thought (CoT) reasoning to agent traces. Add <think>...</think> tags before key decision points to show the agent's reasoning process. The reasoning should explain what information the agent has gathered, what options are available, and why it chose a particular action.

Examples:

Input: "I'll help you track your order. <tool>ask_user(question="What is your order ID?") → 12345</tool> <tool>get_order_details(order_id="12345") → {"status": "delivered", "items": [{"name": "T-shirt"}]}</tool> Your order has been delivered."

Output: "I'll help you track your order. <tool>ask_user(question="What is your order ID?") → 12345</tool> <think>Now that I have the order ID, I should look up the order details to see its status and contents before proceeding.</think> <tool>get_order_details(order_id="12345") → {"status": "delivered", "items": [{"name": "T-shirt"}]}</tool> <think>The order details show it has been delivered, so I should inform the user of this status.</think> Your order has been delivered."

Input: "<tool>ask_user(question="Could you provide your email or name and zip code?") → name: John Smith, zip: 90210</tool> <tool>find_user_id(name="John Smith", zip="90210") → user123</tool> <tool>get_user_orders(user_id="user123") → {"orders": [{"id": "A123", "status": "pending"}, {"id": "B456", "status": "delivered"}]}</tool>"

Output: "<tool>ask_user(question="Could you provide your email or name and zip code?") → name: John Smith, zip: 90210</tool> <think>I need to authenticate the user before proceeding. I'll use their name and zip code to find their user ID.</think> <tool>find_user_id(name="John Smith", zip="90210") → user123</tool> <think>Now that I have authenticated the user, I should check their order history to understand what orders we can discuss.</think> <tool>get_user_orders(user_id="user123") → {"orders": [{"id": "A123", "status": "pending"}, {"id": "B456", "status": "delivered"}]}</tool>"

Add reasoning that demonstrates:
1. Understanding of the current context and available information
2. Evaluation of possible next actions
3. Justification for the chosen action
4. Consideration of policy constraints and requirements
5. Planning multi-step sequences when needed
6. Self-reflection and correction of the previous approach when needed

Keep the reasoning concise but informative. Focus on key decision points where the agent must choose between different actions or interpret information to decide what to do next. Make sure to not accidentally cut parts of the agent trace."""

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
        task["output"] = await add_cot_reasoning(task["output"])
        async with file_lock:
            async with aiofiles.open(output_file, 'a') as f:
                await f.write(json.dumps(task) + '\n')
    return task

async def process_file(input_path: str, output_path: str, max_concurrent: int):
    data = []
    with open(input_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    # Clear the output file
    open(output_path, 'w').close()
    
    semaphore = asyncio.Semaphore(max_concurrent)
    file_lock = asyncio.Lock()
    
    tasks = [process_task(task, semaphore, file_lock, output_path) for task in data]
    await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="cot_converted.json")
    parser.add_argument("--max-concurrent", type=int, default=30, help="Maximum number of concurrent requests")
    args = parser.parse_args()
    
    asyncio.run(process_file(args.input, args.output, args.max_concurrent))

if __name__ == "__main__":
    main()