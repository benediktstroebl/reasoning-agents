import json
import re
from typing import List, Dict, Any
import argparse

def clean_text(text: str) -> str:
   return re.sub(r'\s+', ' ', text).strip()

def convert_function_call(tool_call: Dict[str, Any]) -> str:
   name = tool_call["function"]["name"]
   args = json.loads(tool_call["function"]["arguments"])
   args_str = ", ".join(f"{k}=\"{v}\"" for k,v in args.items())
   response = tool_call.get("response", "")
   return f"<tool>{name}({args_str}) → {response}</tool>"

def process_trajectory(traj: List[Dict[str, Any]]) -> Dict[str, str]:
   system_prompt = next(msg["content"] for msg in traj if msg["role"] == "system")
   output_parts = []
   instruction = ""
   
   first_user_msg = True
   for msg in traj:
       if msg["role"] == "system":
           continue
           
       if msg["role"] == "user":
           if first_user_msg:
               instruction = clean_text(msg['content'])
               first_user_msg = False
           else:
               output_parts.append(f"<user>{clean_text(msg['content'])}</user>")
           continue
           
       if msg["role"] == "assistant":
           if msg["content"]:
               output_parts.append(clean_text(msg["content"]))
               
           if msg.get("tool_calls"):
               for tool_call in msg["tool_calls"]:
                   # Find corresponding tool response
                   tool_response = next((m["content"] for m in traj 
                                      if m["role"] == "tool" and 
                                      m["tool_call_id"] == tool_call["id"]), "")
                   
                   output_parts.append(convert_function_call({
                       "function": tool_call["function"],
                       "response": tool_response
                   }))
   
   return {
       "system_prompt": system_prompt,
       "instruction": instruction,
       "output": " ".join(output_parts)
   }

def convert_logs_to_training_data(logs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
   training_data = []
   
   for log in logs:
       if log["reward"] == 1.0 and "traj" in log:
           try:
               example = process_trajectory(log["traj"])
               training_data.append(example)
           except Exception as e:
               print(f"Error processing trajectory: {e}")
               
   return training_data

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=str, required=True, help="Path to input log file")
   parser.add_argument("--output", type=str, default="training_data.json", help="Path to output training data file")
   args = parser.parse_args()

   with open(args.input) as f:
       logs = json.load(f)
       
   training_data = convert_logs_to_training_data(logs)

   with open(args.output, "w") as f:
       json.dump(training_data, f, indent=2)

if __name__ == "__main__":
   main()