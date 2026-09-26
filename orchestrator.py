"""
Brutalist Multi-Agent Orchestrator
"""
import logging
import os
import subprocess
import textwrap
from typing import Optional

from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Orchestrator:
    """Manages multi-agent workflow to refine LLM evaluation scripts."""
    
    def __init__(self, target_file: str, model: str = "gpt-4o-mini", max_loops: int = 3):
        self.client = OpenAI()
        self.model = model
        self.max_loops = max_loops
        self.target_file = target_file
        self.sandbox_file = target_file.replace(".py", "_sandbox.py")

    def call_agent(self, role: str, system_prompt: str, user_prompt: str) -> str:
        """Interacts with the specified AI agent."""
        print(f"\n[+] {role} is thinking...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Agent {role} failed to respond: {e}")
            return ""

    def run_subprocess(self, cmd: str) -> str:
        """Executes a terminal command and returns its logs."""
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=15)
            return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return "CRASH: Process timed out."
        except Exception as e:
            return f"CRASH: {str(e)}"

    def get_current_code(self) -> str:
        """Reads the core target logic file."""
        try:
            with open(self.target_file, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Target file {self.target_file} not found.")
            return ""

    def save_sandbox(self, code: str) -> None:
        """Writes the proposed code to a sandbox file for testing."""
        with open(self.sandbox_file, "w") as f:
            f.write(code)

    def overwrite_target(self, code: str) -> None:
        """Applies the approved code to the target file."""
        with open(self.target_file, "w") as f:
            f.write(code)

    def execute_workflow(self) -> None:
        """Runs the orchestrator agent loops."""
        current_code = self.get_current_code()
        if not current_code:
            return

        feedback = "Initial check."
        
        for i in range(self.max_loops):
            print(f"\n{'='*50}\n[ITERATION {i+1}/{self.max_loops}]\n{'='*50}")
            
            # 1. STRATEGIST
            sys_strategist = (
                "You are a brutal Data Scientist. "
                "Find weaknesses in the LLM analysis script. "
                "Output ONE concrete paragraph of RFI (Room for Improvement)."
            )
            rfi = self.call_agent("STRATEGIST", sys_strategist, f"Current Code:\n{current_code}\n\nFeedback:\n{feedback}")
            print(f"RFI:\n{rfi}")

            # 2. CODER
            sys_coder = (
                "You are a Senior Python Engineer. "
                "Apply the RFI and rewrite the Python code. "
                "ONLY output the final Python code snippet, without commentary."
            )
            new_code = self.call_agent("CODER (EXECUTION)", sys_coder, f"RFI:\n{rfi}\n\nCurrent Code:\n{current_code}")
            
            if "```python" in new_code:
                new_code = new_code.split("```python")[1].split("```")[0].strip()
            elif "```" in new_code:
                new_code = new_code.split("```")[1].strip()
            
            self.save_sandbox(new_code)
            print(f"[+] Code updated in {self.sandbox_file}")

            # 3. QA
            print("[+] QA is running the tests (Subprocess)...")
            test_logs = self.run_subprocess(f"python {self.sandbox_file}")
            
            sys_qa = (
                "You are a strict QA Agent. "
                "Review the terminal logs. "
                "If there is an error, REJECT. If passed completely, type PASS."
            )
            qa_result = self.call_agent("QA (TESTER)", sys_qa, f"Execution Logs:\n{test_logs}")
            print(f"QA RESULT:\n{qa_result}")
            
            if "REJECT" in qa_result.upper() or "ERROR" in test_logs.upper():
                feedback = f"QA Rejected. Fix code. Reason: {qa_result}\nLogs: {test_logs}"
                continue

            # 4. C-LEVEL
            sys_c_level = (
                "You are a C-Level Judge. "
                "Check if this code is Portfolio Worthy. "
                "If satisfied, MUST answer 'APPROVED'. If bad, reject with reasons."
            )
            c_level_verdict = self.call_agent("C-LEVEL", sys_c_level, f"Code:\n{new_code}\n\nLogs:\n{test_logs}")
            print(f"C-LEVEL VERDICT:\n{c_level_verdict}")
            
            if "APPROVED" in c_level_verdict.upper():
                self.overwrite_target(new_code)
                print(f"\n[!!!] C-LEVEL APPROVED. {self.target_file} OFFICIALLY OVERWRITTEN.")
                break
            else:
                feedback = f"C-Level rejected: {c_level_verdict}"
                print("\n[!] C-Level rejected. Looping back to Strategist...")


if __name__ == "__main__":
    target = os.path.join(os.path.dirname(__file__), "evaluate.py")
    orchestrator = Orchestrator(target_file=target)
    orchestrator.execute_workflow()
