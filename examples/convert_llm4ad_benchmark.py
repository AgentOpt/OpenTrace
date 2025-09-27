#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''convert_llm4ad_benchmark.py
Convert LLM4AD tasks into fully benchmark Trace-ready wrappers.

Unlike the previous version, this creates completely self-contained task modules that:
1. Don't reference the original LLM4AD codebase 
2. Include all necessary evaluation code and data generation
3. Have no hardcoded paths
4. Work without any external dependencies beyond standard libraries + numpy

Each benchmark wrapper exposes:
    build_trace_problem() -> dict

Usage:
    python convert_llm4ad_benchmark.py --llm4ad-root /path/to/LLM4AD --out ./benchmark_tasks
'''

import argparse, sys, os, inspect, importlib, json, shutil
from pathlib import Path
import re
import textwrap
import ast
import runpy

# ------------------------------- Helpers -------------------------------

def read_file(p: Path) -> str:
    try:
        return p.read_text(encoding='utf-8')
    except Exception:
        return ''

def extract_template_program(text: str) -> str | None:
    '''Pull out the Python code inside a variable named `template_program`.'''
    # Try triple-single quotes
    m1 = re.search(r"""template_program\s*=\s*'''(.*?)'''""", text, re.DOTALL)
    if m1:
        return m1.group(1).strip()
    # Try triple-double quotes
    m2 = re.search(r'"""template_program\s*=\s*"""(.*?)""""""', text, re.DOTALL)
    # The above pattern is brittle across snapshots; fallback: generic after '=' until next triple quotes
    m3 = re.search(r'template_program\s*=\s*(?P<q>\"\"\"|\'\'\')(.*?)(?P=q)', text, re.DOTALL)
    if m3:
        return m3.group(2).strip()
    # Fallback: single-line quotes
    m4 = re.search(r"template_program\s*=\s*([\'\"])(.*?)\1", text, re.DOTALL)
    if m4:
        return m4.group(2).strip()
    return None

def extract_task_description(text: str) -> str | None:
    m = re.search(r"task_description\s*=\s*(.+)", text)
    if not m:
        return None
    val = m.group(1).strip()
    if val.startswith(('"', '\'')) and val.endswith(('"', '\'')):
        return val[1:-1]
    return val

def find_entry_function_name(template_code: str) -> str | None:
    '''Find first def name( ... ) in the template code.'''
    m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", template_code, re.MULTILINE)
    return m.group(1) if m else None

def extract_import_header(template_code: str) -> str:
    '''Collect top-of-snippet import lines; ensure numpy/math present.'''
    header_lines = []
    for line in template_code.splitlines():
        s = line.strip()
        if s.startswith('import ') or s.startswith('from '):
            header_lines.append(line.rstrip())
    defaults = ['import numpy as np', 'import math']
    for d in defaults:
        if not any(l.strip().startswith(d) for l in header_lines):
            header_lines.append(d)
    return '\n'.join(header_lines)

def snake_from_parts(parts):
    s = '_'.join(p for p in parts if p)
    s = re.sub(r'[^A-Za-z0-9_]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'task'

def rewrite_imports_for_autonomy(code: str, template_program: str, task_description: str) -> str:
    """Rewrite imports to work with benchmark task structure."""
    lines = []
    template_vars_inserted = False
    path_setup_inserted = False
    
    for line in code.splitlines():
        stripped = line.strip()
        
        # Handle template imports FIRST (before removing llm4ad imports)
        if ('template import template_program' in stripped or 
              'from template import' in stripped):
            # Replace with embedded template values
            lines.append('# ' + line + '  # Template values embedded below')
            if not template_vars_inserted:
                lines.append('')
                lines.append('# Embedded template values')
                lines.append('template_program = ' + repr(template_program))
                lines.append('task_description = ' + repr(task_description))
                lines.append('')
                template_vars_inserted = True
        # Replace LLM4AD base imports
        elif 'from llm4ad.base import Evaluation' in line:
            lines.append('from llm4ad_loader import Evaluation')
        elif stripped.startswith('from llm4ad.') or stripped.startswith('import llm4ad.'):
            # Convert llm4ad imports - utilities to llm4ad_loader, others to local imports
            if 'from llm4ad.task.' in stripped and 'import ' in stripped:
                # Extract the module and imports
                parts = stripped.split(' import ')
                if len(parts) == 2:
                    module_path = parts[0].replace('from ', '')
                    imports = parts[1]
                    
                    # Check if this is a common utility that should come from llm4ad_loader
                    common_utils = ['load_subdir_as_text', 'load_subdir_as_pickle']
                    imported_items = [item.strip() for item in imports.split(',')]
                    
                    # If any imported item is a common utility, import from llm4ad_loader
                    if any(item in common_utils for item in imported_items):
                        # Split into common utilities and local imports
                        loader_imports = [item for item in imported_items if item in common_utils]
                        local_imports = [item for item in imported_items if item not in common_utils]
                        
                        # Add import from llm4ad_loader for utilities
                        if loader_imports:
                            lines.append(f"from llm4ad_loader import {', '.join(loader_imports)}")
                            lines.append('# ' + line + '  # Common utilities from llm4ad_loader')
                        
                        # Add local imports if any remain
                        if local_imports:
                            if not path_setup_inserted:
                                lines.append('import os, sys')
                                lines.append('sys.path.insert(0, os.path.dirname(__file__))')
                                path_setup_inserted = True
                            module_file = module_path.split('.')[-1]
                            lines.append(f"from {module_file} import {', '.join(local_imports)}")
                            lines.append('# ' + line + '  # Local imports converted')
                    else:
                        # Regular local import conversion
                        if not path_setup_inserted:
                            lines.append('import os, sys')
                            lines.append('sys.path.insert(0, os.path.dirname(__file__))')
                            path_setup_inserted = True
                        module_file = module_path.split('.')[-1]
                        new_import = f"from {module_file} import {imports}"
                        lines.append(new_import)
                        lines.append('# ' + line + '  # Converted from LLM4AD import')
                else:
                    lines.append('# ' + line + '  # Removed LLM4AD dependency - using local copies')
            else:
                lines.append('# ' + line + '  # Removed LLM4AD dependency - using local copies')
        elif (stripped.startswith('from ') and 'import ' in stripped and 
              not stripped.startswith('from typing') and
              not stripped.startswith('from __future__') and
              not stripped.startswith('from collections') and
              not stripped.startswith('from itertools') and
              not stripped.startswith('from functools') and
              not stripped.startswith('from math') and
              not stripped.startswith('from numpy') and
              not stripped.startswith('from llm4ad_loader') and
              not '.' in stripped.split()[1]):  # Local import (no dots)
            # This is likely a local import - add path setup
            if not path_setup_inserted:
                lines.append('import os, sys')
                lines.append('sys.path.insert(0, os.path.dirname(__file__))')
                path_setup_inserted = True
            lines.append(line)
        elif (stripped.startswith('import ') and 
              not stripped.startswith('import numpy') and 
              not stripped.startswith('import math') and
              not stripped.startswith('import os') and
              not stripped.startswith('import sys') and
              not stripped.startswith('import itertools') and
              not stripped.startswith('import random') and
              not stripped.startswith('import json') and
              not stripped.startswith('import pickle') and
              not '.' in stripped.split()[1]):  # Local import (no dots)
            # This might be a local import - add path setup
            if not path_setup_inserted:
                lines.append('import os, sys')
                lines.append('sys.path.insert(0, os.path.dirname(__file__))')
                path_setup_inserted = True
            lines.append(line)
        else:
            lines.append(line)
    
    return '\n'.join(lines)

def extract_evaluation_class(evaluation_file: Path) -> tuple[str, str]:
    """Extract the evaluation class name and its full code."""
    content = read_file(evaluation_file)
    
    # Find the evaluation class definition
    class_match = re.search(r'class\s+([A-Za-z_]\w*)\(Evaluation\)', content)
    if not class_match:
        raise ValueError(f"No Evaluation subclass found in {evaluation_file}")
    
    class_name = class_match.group(1)
    
    return class_name, content

# ------------------------------- Core ----------------------------------

def discover_task_pairs(llm4ad_root: Path, requested_filters: list[str] | None):
    '''Yield (template_path, evaluation_path, family_key).'''
    candidates = []
    # example/*
    ex = llm4ad_root / 'example'
    if ex.exists():
        for tpl in ex.rglob('template.py'):
            fam = tpl.parent
            ev = fam / 'evaluation.py'
            if ev.exists():
                rel = tpl.relative_to(ex)
                key = rel.parts[0] if len(rel.parts)>0 else rel.stem
                candidates.append((tpl, ev, key))
    # llm4ad/task/*
    task_root = llm4ad_root / 'llm4ad' / 'task'
    if task_root.exists():
        for tpl in task_root.rglob('template.py'):
            fam = tpl.parent
            ev = fam / 'evaluation.py'
            if ev.exists():
                rel = tpl.relative_to(task_root)
                # Use the full relative path without the template.py part for unique keys
                key = '/'.join(rel.parts[:-1]) if len(rel.parts) > 1 else rel.stem
                candidates.append((tpl, ev, key))
    # filter & dedup
    pairs, seen = [], set()
    for tpl, ev, key in candidates:
        h = (str(tpl), str(ev))
        if h in seen:
            continue
        seen.add(h)
        if requested_filters:
            if not any(f in str(tpl) or f in str(ev) or f in key for f in requested_filters):
                continue
        pairs.append((tpl, ev, key))
    return pairs

def copy_task_dependencies(task_dir: Path, out_task_dir: Path) -> list[str]:
    """Copy additional files needed by a task (e.g., data generators)."""
    copied_files = []
    
    # Copy all Python files except template.py and evaluation.py
    for py_file in task_dir.glob('*.py'):
        if py_file.name not in ('template.py', 'evaluation.py'):
            dest = out_task_dir / py_file.name
            shutil.copy2(py_file, dest)
            copied_files.append(py_file.name)
    
    # Copy paras.yaml if it exists
    paras_file = task_dir / 'paras.yaml'
    if paras_file.exists():
        shutil.copy2(paras_file, out_task_dir / 'paras.yaml')
        copied_files.append('paras.yaml')
    
    # Copy any data files or other resources
    for ext in ['*.txt', '*.json', '*.csv', '*.dat']:
        for data_file in task_dir.glob(ext):
            dest = out_task_dir / data_file.name
            shutil.copy2(data_file, dest)
            copied_files.append(data_file.name)
    
    return copied_files

WRAPPER_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous LLM4AD task: {task_name}
Generated by convert_llm4ad_benchmark.py

This is a fully self-contained task module that doesn't depend on the original LLM4AD codebase.
"""

# Embedded evaluation code (benchmark)
{evaluation_code}

# Task configuration for benchmark task
ENTRY_NAME = {entry_name!r}
FUNCTION_SIGNATURE = {function_signature!r}
IMPORT_HEADER = {import_header!r}
TASK_DESCRIPTION = {task_description!r}
OBJECTIVE_TEXT = {objective_text!r}
TEMPLATE_FUNCTION = {template_function!r}
EVAL_CLASS_NAME = {eval_class_name!r}
EVAL_KWARGS = {eval_kwargs!r}

def build_trace_problem(**override_eval_kwargs) -> dict:
    """Build a Trace-ready problem using embedded benchmark evaluator."""
    
    # Create evaluator instance with embedded class
    eval_kwargs_final = EVAL_KWARGS.copy()
    eval_kwargs_final.update(override_eval_kwargs)
    
    evaluator = globals()[EVAL_CLASS_NAME](**eval_kwargs_final)
    
    from llm4ad_loader import AutonomousEvaluatorGuide
    from opto import trace
    
    # Create parameter
    initial_code = TEMPLATE_FUNCTION.strip()
    param = trace.node(initial_code, name='__code', 
                      description=f'The code should start with: {{FUNCTION_SIGNATURE}}', 
                      trainable=True)
    
    # Create guide using benchmark embedded evaluator
    guide = AutonomousEvaluatorGuide(evaluator, ENTRY_NAME, IMPORT_HEADER, 
                                   timeout=eval_kwargs_final.get('timeout_seconds', 30))
    
    # Create dataset
    train_dataset = dict(
        inputs=[TASK_DESCRIPTION],
        infos=[{{'imports': IMPORT_HEADER, 'entry': ENTRY_NAME}}]
    )
    
    # Optimizer kwargs
    optimizer_kwargs = dict(
        objective=OBJECTIVE_TEXT,
        memory_size=10
    )
    
    return dict(
        param=param,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(
            entry=ENTRY_NAME,
            function_signature=FUNCTION_SIGNATURE,
            eval_class=EVAL_CLASS_NAME,
            benchmark=True,
        )
    )
'''

def main():
    ap = argparse.ArgumentParser(description='Convert LLM4AD tasks into benchmark Trace wrappers.')
    ap.add_argument('--llm4ad-root', type=str, required=True, help='Path to LLM4AD repository root.')
    ap.add_argument('--out', type=str, default='./benchmark_tasks', help='Output folder for benchmark task modules.')
    ap.add_argument('--select', type=str, default='', help='Comma-separated substrings to filter tasks.')
    args = ap.parse_args()

    llm4ad_root = Path(args.llm4ad_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    filters = [s.strip() for s in args.select.split(',') if s.strip()] if args.select else None

    pairs = discover_task_pairs(llm4ad_root, filters)

    if not pairs:
        print('No (template.py, evaluation.py) pairs found with current filters.')
        sys.exit(1)

    index = []

    for tpl, ev, fam_key in pairs:
        try:
            tpl_txt = read_file(tpl)
            ev_txt = read_file(ev)

            template_code = extract_template_program(tpl_txt)
            if not template_code:
                print(f'[SKIP] Could not extract template_program from {tpl}')
                continue

            entry = find_entry_function_name(template_code)
            if not entry:
                print(f'[SKIP] Could not find entry function in template_program at {tpl}')
                continue

            # description
            task_desc = extract_task_description(tpl_txt) or f'Implement {entry}() to solve the problem.'

            # Extract evaluation class with template values
            eval_class_name, eval_code = extract_evaluation_class(ev)
            eval_code = rewrite_imports_for_autonomy(eval_code, template_code, task_desc)
            
            imports = extract_import_header(template_code)
            # Capture function signature for clarity
            fsig = re.search(r'(^\s*def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*:\s*)', template_code, re.MULTILINE)
            fsig_str = fsig.group(1).strip() if fsig else f'def {entry}(...):'

            objective_text = (f"You are optimizing the implementation of `{entry}` for the LLM4AD task.\\n\\n"
                              f"Task description:\\n{task_desc}\\n\\n"
                              f"Your goal is to return a correct and efficient function whose score (computed by the task evaluator) is as high as possible.")

            # file name - use full path to avoid collisions
            parts = fam_key.split('/')
            if len(parts) >= 3 and parts[0] == 'optimization' and parts[1] == 'co_bench':
                task_name = parts[2].replace('_co_bench', '') if parts[2].endswith('_co_bench') else parts[2]
                short_key = snake_from_parts([parts[0], task_name])
            elif len(parts) >= 3:
                short_key = snake_from_parts(parts[:3])
            else:
                short_key = snake_from_parts(parts[:2])
            mod_name = short_key if short_key else snake_from_parts([entry])
            
            # Create task directory
            task_dir = out / mod_name
            task_dir.mkdir(exist_ok=True)
            
            # Copy task dependencies
            copied_files = copy_task_dependencies(ev.parent, task_dir)

            # Load eval kwargs from paras.yaml
            paras_yaml = ev.parent / 'paras.yaml'
            eval_kwargs = {}
            if paras_yaml.exists():
                try:
                    import yaml  # optional
                    eval_kwargs = yaml.safe_load(paras_yaml.read_text())
                    if isinstance(eval_kwargs, dict):
                        eval_kwargs.pop('name', None)
                except Exception:
                    eval_kwargs = {}

            # Create benchmark wrapper
            wrapper_content = WRAPPER_TEMPLATE.format(
                task_name=mod_name,
                evaluation_code=eval_code,
                entry_name=entry,
                function_signature=fsig_str,
                import_header=imports,
                task_description=task_desc,
                objective_text=objective_text,
                template_function=template_code,
                eval_class_name=eval_class_name,
                eval_kwargs=eval_kwargs
            )
            
            wrapper_path = task_dir / '__init__.py'
            wrapper_path.write_text(wrapper_content, encoding='utf-8')
            
            index.append(dict(
                key=fam_key,
                module=str(task_dir.relative_to(out)),
                entry=entry,
                eval_class=eval_class_name,
                task_description=task_desc,
                wrapper=mod_name,
                copied_files=copied_files,
                benchmark=True
            ))
            print(f"[OK] Created benchmark task {task_dir}")
            
        except Exception as e:
            print(f"[ERROR] Failed to convert {fam_key}: {e}")
            continue

    (out / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    print(f"\\nCreated {len(index)} benchmark tasks at {out}")

if __name__ == '__main__':
    main()