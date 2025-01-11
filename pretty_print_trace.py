import json
import re
from typing import List, Dict

def extract_tool_calls(text: str) -> List[Dict]:
    """Extract tool calls and their responses from text."""
    pattern = r'<tool>(.*?)\) \u2192 (.*?)</tool>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    tool_calls = []
    for call, response in matches:
        tool_calls.append({
            "call": call.strip(),
            "response": response.strip()
        })
    return tool_calls

def extract_thoughts(text: str) -> List[str]:
    """Extract thoughts from text."""
    pattern = r'<think>(.*?)</think>'
    return re.findall(pattern, text, re.DOTALL)

def pretty_print_trace(instruction: str, trace: str):
    """Pretty print an agent trace showing tools, thoughts and outputs."""
    # Print instruction
    print("\n=== USER INSTRUCTION ===")
    print(instruction.strip())
    
    # Extract tool calls and thoughts
    tool_calls = extract_tool_calls(trace)
    thoughts = extract_thoughts(trace)
    
    # Print sequence
    step = 1
    thought_idx = 0
    
    for tool in tool_calls:
        # Print any thoughts before this tool call
        while thought_idx < len(thoughts) and thoughts[thought_idx] in trace.split(tool["call"])[0]:
            print(f"\n=== Step {step}: THOUGHT ===")
            print(thoughts[thought_idx].strip())
            step += 1
            thought_idx += 1
            
        print(f"\n=== Step {step}: TOOL CALL ===")
        print(f"Call: {tool['call']}")
        print(f"Response: {tool['response']}")
        step += 1
    
    # Print any remaining thoughts
    while thought_idx < len(thoughts):
        print(f"\n=== Step {step}: THOUGHT ===")
        print(thoughts[thought_idx].strip())
        step += 1
        thought_idx += 1

if __name__ == "__main__":
    # Read the trace file
    with open("cot_converted.json") as f:
        data = json.load(f)
        
    # Pretty print each trace
    for i, item in enumerate(data):
        print(f"\n\n{'='*80}")
        print(f"TRACE {i+1}")
        print('='*80)
        pretty_print_trace(item["instruction"], item["output"]) 