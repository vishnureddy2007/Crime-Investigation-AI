"""
AI Health Check Service.
Provides real-time diagnostics for the Ollama/Qwen pipeline.
"""

from __future__ import annotations
import requests
import json
from typing import TypedDict
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

class HealthStatus(TypedDict):
    provider: str
    model: str
    connection: str  # PASS/FAIL
    model_available: str # PASS/FAIL
    generation: str # PASS/FAIL
    parsing: str # PASS/FAIL
    details: str

class AIHealthChecker:
    """
    Performs a live test of the AI pipeline to ensure it is operational.
    """
    
    def check_all(self) -> HealthStatus:
        status: HealthStatus = {
            "provider": "Ollama",
            "model": OLLAMA_MODEL,
            "connection": "FAIL",
            "model_available": "FAIL",
            "generation": "FAIL",
            "parsing": "FAIL",
            "details": ""
        }
        
        try:
            # 1. Connection Test
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                status["details"] = f"Ollama connection failed (Status: {resp.status_code})"
                return status
            status["connection"] = "PASS"
            
            # 2. Model Availability Test
            tags = resp.json().get("models", [])
            model_names = [m['name'] for m in tags]
            if OLLAMA_MODEL not in model_names:
                status["details"] = f"Model {OLLAMA_MODEL} not found in Ollama. Available: {model_names}"
                return status
            status["model_available"] = "PASS"
            
            # 3. Generation & Parsing Test
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": "Respond with JSON: {'status': 'ok'}",
                "stream": False,
                "format": "json"
            }
            gen_resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60.0)
            if gen_resp.status_code == 200:
                status["generation"] = "PASS"
                try:
                    res_data = gen_resp.json().get("response", "")
                    json.loads(res_data)
                    status["parsing"] = "PASS"
                except Exception as e:
                    status["details"] = f"Parsing failed: {e}"
            else:
                status["details"] = f"Generation failed (Status: {gen_resp.status_code})"
                
        except Exception as e:
            status["details"] = f"Health check exception: {str(e)}"
            
        return status

def get_ai_health() -> HealthStatus:
    return AIHealthChecker().check_all()
