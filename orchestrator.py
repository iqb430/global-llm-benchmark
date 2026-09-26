"""
Brutalist Multi-Agent Orchestrator
"""
import logging
import os
import subprocess
import textwrap
from typing import Optional

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"

class Orchestrator:
    """Manages multi-agent workflow to refine LLM evaluation scripts."""
    
    for i in range(MAX_LOOPS):
        print(f"\n{'='*50}\n[ITERATION {i+1}/{MAX_LOOPS}]\n{'='*50}")
        
        sys_strategist = "Lu Data Scientist brutal. Cari kelemahan script analisis LLM. Output SATU paragraf RFI (Room for Improvement) konkrit."
        rfi = call_agent("STRATEGIST", sys_strategist, f"Code saat ini:\n{current_code}\n\nFeedback iterasi kemaren:\n{feedback}")
        print(f"RFI:\n{rfi}")

        sys_coder = "Lu Senior Python Engineer. Berdasarkan RFI, ubah/tulis ulang kode Python-nya. HANYA OUTPUT FULL CODE [PYTHON], gak usah bacot. Jangan pake em-dash."
        new_code = call_agent("CODER (EXECUTION)", sys_coder, f"RFI:\n{rfi}\n\nCode lawas:\n{current_code}")
        if "```python" in new_code:
            new_code = new_code.split("```python")[1].split("```")[0].strip()
        
        with open("evaluate_sandbox.py", "w") as f:
            f.write(new_code)
            
            # 1. STRATEGIST
            sys_strategist = (
                "You are a brutal Data Scientist. "
                "Find weaknesses in the LLM analysis script. "
                "Output ONE concrete paragraph of RFI (Room for Improvement)."
            )
            rfi = self.call_agent("STRATEGIST", sys_strategist, f"Current Code:\n{current_code}\n\nFeedback:\n{feedback}")
            print(f"RFI:\n{rfi}")

        print("[+] QA ngetes jalanin codenya (Subprocess)...")
        test_logs = run_subprocess("python evaluate_sandbox.py")
        sys_qa = "Lu QA Agent galak. Benci filler AI. Periksa log terminal hasil eksekusi code. Kalo error atau bahasanya cringe/AI banget, REJECT. Kalo mulus, ketik PASS."
        qa_result = call_agent("QA (TESTER)", sys_qa, f"Execution Logs:\n{test_logs}")
        print(f"QA RESULT:\n{qa_result}")
        
        if "REJECT" in qa_result.upper() or "ERROR" in test_logs:
            feedback = f"QA Nolak. Benerin codenya. Alasan QA: {qa_result}\nLogs: {test_logs}"
            continue

        sys_c_level = "Lu C-Level Judge. Style lu Dostoevsky x BMO. Cek apa code dan hasil ini 'Portfolio Worthy' (Brutalist, mekanik kuat, gak murahan). Kalo puas, jawab mutlak 'APPROVED'. Kalo sampah, reject dengan alasan detail."
        c_level_verdict = call_agent("C-LEVEL", sys_c_level, f"Code:\n{new_code}\n\nExecution Logs:\n{test_logs}")
        print(f"C-LEVEL VERDICT:\n{c_level_verdict}")
        
        if "APPROVED" in c_level_verdict:
            with open("evaluate.py", "w") as f:
                f.write(new_code)
            print("\n[!!!] C-LEVEL APPROVED. evaluating.py RESMI DI-OVERWRITE.")
            break
        else:
            feedback = f"C-Level nge-reject: {c_level_verdict}"
            print("\n[!] Di-reject C-Level. Looping balik ke Strategist...")

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(__file__), "evaluate.py")
    orchestrator = Orchestrator(target_file=target)
    orchestrator.execute_workflow()
