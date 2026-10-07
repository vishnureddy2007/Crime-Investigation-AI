"""
Pipeline Performance Profiler for Crime Investigation AI.
Measures execution latency, RAM RSS, CPU %, and VRAM for every stage in the workflow.
"""

from __future__ import annotations

import time
import os
import sys
import json
from pathlib import Path
import psutil
import torch

# Add workspace to path
ws_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ws_dir))

def get_process_stats():
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    ram_mb = round(mem_info.rss / (1024 * 1024), 2)
    cpu_pct = psutil.cpu_percent(interval=None)
    
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
    
    return ram_mb, cpu_pct, vram_mb

def profile():
    report = []
    print("=" * 60)
    print("STARTING FULL PIPELINE PERFORMANCE PROFILING")
    print("=" * 60)

    # 1. App startup / Imports
    t0 = time.perf_counter()
    import config
    from database.db import init_db
    from database.repository import save_analysis, save_case
    from models.yolo_detector import get_multi_source_detector
    from models.evidence_analyzer import analyze, AnalysisInput
    from models.schemas import AnalysisInput
    from services.crime_timeline import build_timeline
    from models.summary_generator import SummaryGenerator
    from services.crime_situation import CrimeSituationAnalyzer
    from services.chat_assistant import ChatAssistant
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("Application Imports & Setup", round(t1 - t0, 3), ram, cpu, vram))

    # 2. Database Init
    t0 = time.perf_counter()
    init_db(config.DATABASE_PATH)
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("Database Initialization", round(t1 - t0, 3), ram, cpu, vram))

    # 3. YOLO Model Loading
    t0 = time.perf_counter()
    detector = get_multi_source_detector()
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("YOLO Multi-Source Load", round(t1 - t0, 3), ram, cpu, vram))

    # 4. Sample Image Load & Preprocessing
    sample_img_path = ws_dir / "tests" / "fixtures" / "sample.jpg"
    t0 = time.perf_counter()
    from PIL import Image
    img = Image.open(sample_img_path)
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("Image Load & Preprocess", round(t1 - t0, 3), ram, cpu, vram))

    # 5. YOLO Inference
    t0 = time.perf_counter()
    det_result = detector.detect_image(img, source_name="sample.jpg")
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("YOLO Multi-Pass Inference", round(t1 - t0, 3), ram, cpu, vram))

    # 6. Weapon Filtering & HITL Prep
    t0 = time.perf_counter()
    input_data = AnalysisInput.from_image(det_result)
    analysis = analyze(input_data)
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("Weapon Filtering & Evidence Prep", round(t1 - t0, 3), ram, cpu, vram))

    # 7. Timeline Generation
    t0 = time.perf_counter()
    timeline = build_timeline(analysis)
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("Timeline Generation", round(t1 - t0, 3), ram, cpu, vram))

    # 8. Ollama Connection & Health Check
    t0 = time.perf_counter()
    sum_gen = SummaryGenerator()
    online = sum_gen.is_available()
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append((f"Ollama Reachability Check (Online={online})", round(t1 - t0, 3), ram, cpu, vram))

    # 9. Qwen3 14B AI Summary Generation
    t0 = time.perf_counter()
    summary_obj = sum_gen.generate(analysis)
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("Qwen3 14B Summary Generation", round(t1 - t0, 3), ram, cpu, vram))

    # 10. Qwen3 14B Situation Prediction
    t0 = time.perf_counter()
    sit_analyzer = CrimeSituationAnalyzer()
    situation_obj = sit_analyzer.analyze(analysis, summary_text=summary_obj.investigation_summary)
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("Qwen3 14B Situation Prediction", round(t1 - t0, 3), ram, cpu, vram))

    # 11. Chat Assistant Query
    t0 = time.perf_counter()
    chat = ChatAssistant(analysis=analysis)
    reply = chat.reply("What weapons were identified?")
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("Qwen3 14B Chat Assistant Response", round(t1 - t0, 3), ram, cpu, vram))

    # 12. Database Save
    t0 = time.perf_counter()
    case_id = save_case(config.DATABASE_PATH, "sample.jpg", "image")
    save_analysis(config.DATABASE_PATH, case_id, analysis)
    t1 = time.perf_counter()
    ram, cpu, vram = get_process_stats()
    report.append(("Database Save", round(t1 - t0, 3), ram, cpu, vram))

    print("\n" + "=" * 70)
    print(f"{'STAGE':<35} | {'TIME (s)':<8} | {'RAM (MB)':<9} | {'CPU %':<6} | {'VRAM (MB)'}")
    print("=" * 70)
    tot_time = 0.0
    for name, dur, ram, cpu, vram in report:
        tot_time += dur
        print(f"{name:<35} | {dur:<8.3f} | {ram:<9.1f} | {cpu:<6.1f} | {vram}")
    print("=" * 70)
    print(f"{'TOTAL PIPELINE TIME':<35} | {tot_time:<8.3f}s")
    print("=" * 70)

if __name__ == "__main__":
    profile()
