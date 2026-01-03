#!/usr/bin/env python3
"""Summarize Jupyter notebooks in a directory using Gemini-Pro (or fallback).

Usage:
  python summarize_notebooks.py --dir Clases --out output.txt

Environment:
  - Set `GEMINI_API_KEY` and `GEMINI_MODEL` for Gemini-Pro calls.
    Example model: models/gemini-2.5-pro
  - If `GEMINI_API_KEY` is not set or `--mock` used, a local fallback summarizer runs.

The script writes concatenated bullet summaries per notebook to the output file.
"""

import os
import json
import argparse
import re
import time
from typing import List

try:
    import google.generativeai as genai
except Exception:
    genai = None


def extract_text_from_notebook(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    parts: List[str] = []
    for cell in nb.get('cells', []):
        ctype = cell.get('cell_type')
        src = ''.join(cell.get('source', []))
        if not src:
            continue
        if ctype == 'markdown':
            parts.append(src)
        elif ctype == 'code':
            # extract comments and docstrings from code cell
            comments = []
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith('#'):
                    comments.append(stripped.lstrip('# ').rstrip())
            # extract triple-quoted strings (simple heuristic)
            docstrings = re.findall(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', src, re.S)
            doctexts = [t[0] or t[1] for t in docstrings if (t[0] or t[1])]
            if comments:
                parts.append('\n'.join(comments))
            if doctexts:
                parts.append('\n'.join(doctexts))

    return '\n\n'.join(parts)


def build_prompt(notebook_name: str, text: str) -> str:
    prompt = (
        f"Resume en español los conceptos principales del cuaderno '{notebook_name}'. "
        "Devuelve una lista de viñetas cortas (cada viñeta 3-10 palabras) con los temas y conceptos importantes. "
        "Prioriza títulos, definiciones, operadores, ejemplos y buenas prácticas. "
        "No incluyas líneas vacías. Limita a 12 viñetas. "
        "Salida: una viñeta por línea que empiece con '-' seguido de un espacio.\n\n"
        "Texto del cuaderno:\n" + text
    )
    return prompt


def call_gemini(prompt: str, model: str, api_key: str, max_tokens: int = 512, timeout: int = 60) -> str:
    """Call Gemini via google.generativeai. Raises if library or key absent."""
    if genai is None:
        raise RuntimeError('google.generativeai library is required for Gemini calls')
    if not api_key:
        raise RuntimeError('API key is required for Gemini calls')

    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)
    # The library returns an object where `.text` is commonly available.
    resp = gm.generate_content(prompt)

    # Try common response attributes
    if hasattr(resp, 'text') and resp.text:
        return resp.text
    if hasattr(resp, 'candidates') and resp.candidates:
        # candidate objects may contain `content` or `output`
        cand = resp.candidates[0]
        return getattr(cand, 'content', None) or getattr(cand, 'output', None) or str(cand)
    if hasattr(resp, 'outputs') and resp.outputs:
        out = resp.outputs[0]
        return getattr(out, 'content', None) or str(out)

    return str(resp)


def fallback_summarize(text: str, max_bullets: int = 10) -> List[str]:
    raise RuntimeError('Fallback summarizer removed: the script must use Gemini and fail on errors.')


def summarize_notebooks(dirpath: str, outpath: str, use_gemini: bool, model: str, api_key: str, mock: bool):
    files = [f for f in os.listdir(dirpath) if f.endswith('.ipynb')]
    files.sort()
    results = []

    def last_processed_index(path: str) -> int:
        if not os.path.exists(path):
            return -1
        mx = -1
        pat = re.compile(r'^Clase\s*(\d+):', re.IGNORECASE)
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    m = pat.match(line.strip())
                    if m:
                        try:
                            v = int(m.group(1))
                            if v > mx:
                                mx = v
                        except Exception:
                            pass
        except Exception:
            return -1
        return mx

    start_idx = last_processed_index(outpath) + 1
    if start_idx > 0:
        print(f'Reanudando desde Clase {start_idx} (según {outpath})')

    for idx, fname in enumerate(files):
        if idx < start_idx:
            print(f'Saltando {fname} (Clase {idx}) — ya procesado')
            continue
        full = os.path.join(dirpath, fname)
        print(f'Procesando {fname}...')
        text = extract_text_from_notebook(full)
        if not text.strip():
            text = ' '  # avoid empty prompt

        # Always call Gemini; on any error the script should fail (no fallback)
        prompt = build_prompt(fname, text)
        out = call_gemini(prompt, model, api_key)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if not lines:
            raise RuntimeError(f'Empty response from model for {fname}')

        header = f'Clase {idx}:'
        # Write immediately to outpath to avoid data loss on crash
        try:
            with open(outpath, 'a', encoding='utf-8') as fh:
                fh.write(header + '\n')
                for l in lines:
                    fh.write(l + '\n')
                fh.write('\n')
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError(f'Error writing to {outpath}: {e}')

        results.append((header, lines))
        # polite pause to avoid rate limits
        time.sleep(0.5)

    # write output
    with open(outpath, 'w', encoding='utf-8') as out:
        for header, lines in results:
            out.write(header + '\n')
            for l in lines:
                out.write(l + '\n')
            out.write('\n')

    print(f'Escrito resumen a {outpath} — {len(results)} notebooks procesados.')


def main():
    parser = argparse.ArgumentParser(description='Summarize notebooks in a directory using Gemini-Pro or fallback.')
    parser.add_argument('--dir', type=str, default='Clases', help='Subdirectory (relative to script) containing .ipynb files')
    parser.add_argument('--out', type=str, default='output.txt', help='Output file name')
    parser.add_argument('--model', type=str, default=os.environ.get('GEMINI_MODEL', 'models/gemini-2.5-pro'), help='Gemini model name')
    parser.add_argument('--mock', action='store_true', help='(DEPRECATED) mock disabled — no fallback allowed')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, args.dir)
    outpath = os.path.join(script_dir, args.out)

    if not os.path.isdir(target_dir):
        print('Directorio no encontrado:', target_dir)
        return

    # Load .env from the target directory (if present) to allow per-folder secrets
    def load_env_file(path: str) -> None:
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    # do not overwrite existing env vars
                    if k and k not in os.environ:
                        os.environ[k] = v
        except FileNotFoundError:
            pass

    env_path = os.path.join(target_dir, '.env')
    load_env_file(env_path)

    # Enforce presence of the generative library and API key. No fallback allowed.
    if args.mock:
        print('--mock flag is not supported: fallback removed. Exiting.')
        return

    if genai is None:
        print('google.generativeai library is not installed. Install with: pip install google-generativeai')
        return

    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print('Environment variable GEMINI_API_KEY or GOOGLE_API_KEY not set. Exiting.')
        return

    # proceed — any failures from the model will raise
    summarize_notebooks(target_dir, outpath, True, args.model, api_key, False)


if __name__ == '__main__':
    main()
