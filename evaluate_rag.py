from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

# ==============================================================================
# 1. CONFIGURACIÓN DEL FLUJO Y MODELOS
# ==============================================================================
OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "http://localhost:3000/api")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "")

# Modelo bajo prueba (tu modelo personalizado con RAG en Open WebUI)
TARGET_MODEL = os.getenv("TARGET_MODEL", "modelo-fitness-colombiano")

# Modelo juez opcional para evaluar fidelidad/completitud (1 a 5).
# Déjalo vacío ("") si solo quieres medir latencia, éxito y palabras clave.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")

# Etiqueta de versión para comparar experimentos (ej: "v1_chunk500_ov50")
RUN_TAG = os.getenv("RUN_TAG", "baseline")

DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "runs"

client = OpenAI(
    base_url=OPENWEBUI_BASE_URL,
    api_key=OPENWEBUI_API_KEY,
)


def load_dataset(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("id", f"case_{line_num:03d}")
            row.setdefault("category", "general")
            row.setdefault("expected_answer", "")
            row.setdefault("required_keywords", [])
            items.append(row)
    return items


def query_target_model(prompt: str) -> tuple[str, float, str | None]:
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        elapsed = time.perf_counter() - t0
        content = (resp.choices[0].message.content or "").strip()
        return content, elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return "", elapsed, str(exc)


def check_required_keywords(answer: str, keywords: list[str]) -> dict[str, bool]:
    lower_ans = answer.lower()
    return {kw: (kw.lower() in lower_ans) for kw in keywords}


def judge_answer(
    prompt: str, expected: str, actual: str
) -> tuple[int | None, str]:
    if not JUDGE_MODEL or not actual:
        return None, "skipped"

    judge_system = (
        "Eres un evaluador estricto de sistemas RAG de fitness y nutrición. "
        "Evalúa ACTUAL frente a EXPECTED para el PROMPT dado.\n"
        "Puntúa de 1 a 5:\n"
        "5 = Cubre los puntos clave, respeta el nivel/restricciones y no contradice lo esperado.\n"
        "3 = Parcialmente correcto o incompleto, pero útil.\n"
        "1 = Incorrecto, fuera de tema o ignora restricciones importantes.\n"
        'Responde ÚNICAMENTE con un JSON válido: {"score": <1-5>, "reason": "<motivo breve>"}'
    )
    user_msg = (
        f"PROMPT:\n{prompt}\n\n"
        f"EXPECTED:\n{expected}\n\n"
        f"ACTUAL:\n{actual}"
    )

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Limpiar posibles bloques ```json ... ```
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        score = int(parsed.get("score", 0))
        if score < 1 or score > 5:
            score = None
        return score, str(parsed.get("reason", ""))
    except Exception as exc:
        return None, f"judge_error: {exc}"


def main() -> None:
    if not OPENWEBUI_API_KEY:
        raise RuntimeError("Falta OPENWEBUI_API_KEY en el entorno o .env")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontró {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"eval_{RUN_TAG}_{timestamp}.jsonl"

    dataset = load_dataset(DATASET_PATH)
    print(
        f"Ejecutando {len(dataset)} casos | Modelo: {TARGET_MODEL} | Tag: {RUN_TAG}"
    )

    records = []
    for idx, case in enumerate(dataset, start=1):
        actual, latency_s, error = query_target_model(case["prompt"])
        kw_results = check_required_keywords(actual, case["required_keywords"])
        kw_passed = all(kw_results.values()) if kw_results else True

        score, judge_reason = judge_answer(
            case["prompt"], case["expected_answer"], actual
        )

        rec = {
            "id": case["id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "expected_answer": case["expected_answer"],
            "actual_answer": actual,
            "latency_s": round(latency_s, 3),
            "error": error,
            "keywords_check": kw_results,
            "keywords_passed": kw_passed,
            "judge_score": score,
            "judge_reason": judge_reason,
        }
        records.append(rec)

        status = "ERR" if error else f"{latency_s:.2f}s"
        score_str = f" | score={score}/5" if score is not None else ""
        print(f"[{idx}/{len(dataset)}] {case['id']} ({status}){score_str}")

    with out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Resumen de métricas
    total = len(records)
    errors = sum(1 for r in records if r["error"])
    latencies = [r["latency_s"] for r in records if not r["error"]]
    kw_rate = (
        sum(1 for r in records if not r["error"] and r["keywords_passed"])
        / max(total - errors, 1)
    ) * 100
    scores = [
        r["judge_score"] for r in records if r["judge_score"] is not None
    ]

    print("\n=== RESUMEN DE EVALUACIÓN ===")
    print(f"Archivo guardado : {out_file}")
    print(f"Casos totales    : {total}")
    print(f"Errores HTTP/API : {errors}")
    if latencies:
        print(
            f"Latencia prom.   : {statistics.mean(latencies):.2f}s | "
            f"Mediana: {statistics.median(latencies):.2f}s"
        )
    print(f"Keywords OK      : {kw_rate:.1f}%")
    if scores:
        print(f"Promedio Juez    : {statistics.mean(scores):.2f} / 5.00")


if __name__ == "__main__":
    main()