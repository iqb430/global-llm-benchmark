import os
import subprocess
import json
import textwrap
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"

def call_agent(role: str, system_prompt: str, user_prompt: str) -> str:
    print(f"\n[+] {role} lagi mikir...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

def run_subprocess(cmd: str) -> str:
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=15)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"CRASH: {str(e)}"

def orchestrate():
    MAX_LOOPS = 3
    current_code = ""
    with open("evaluate.py", "r") as f:
        current_code = f.read()

    feedback = "Initial check."
    
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
            
        print("[+] Code di-update di evaluate_sandbox.py")

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
    orchestrate()
