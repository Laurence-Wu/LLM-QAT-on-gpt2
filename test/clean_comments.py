import ast
import sys
from pathlib import Path

class CommentRemover(ast.NodeTransformer):
    def __init__(self):
        self.lines_removed = 0

    def visit_Module(self, node):
        if ast.get_docstring(node):
            self.lines_removed += len(ast.get_docstring(node).split('\n'))
        node.body = [n for n in node.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)]
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        if ast.get_docstring(node):
            self.lines_removed += len(ast.get_docstring(node).split('\n'))
        node.body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        if ast.get_docstring(node):
            self.lines_removed += len(ast.get_docstring(node).split('\n'))
        node.body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        if ast.get_docstring(node):
            self.lines_removed += len(ast.get_docstring(node).split('\n'))
        node.body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        self.generic_visit(node)
        return node

def remove_comments_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_lines = content.split('\n')

    lines = []
    comment_count = 0

    for line in original_lines:
        stripped = line.strip()
        if stripped.startswith('#!'):
            comment_count += 1
            continue

        if stripped.startswith('#'):
            comment_count += 1
            continue

        if '#' in line and not ('"' in line or "'" in line):
            parts = line.split('#')
            if len(parts) > 1:
                code_part = parts[0].rstrip()
                if code_part:
                    lines.append(code_part)
                    comment_count += 1
                else:
                    comment_count += 1
                continue

        lines.append(line)

    cleaned_content = '\n'.join(lines)

    try:
        tree = ast.parse(cleaned_content)
        remover = CommentRemover()
        new_tree = remover.visit(tree)
        ast.fix_missing_locations(new_tree)
        final_content = ast.unparse(new_tree)
        total_removed = comment_count + remover.lines_removed
        return final_content, total_removed
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return cleaned_content, comment_count

def clean_file(file_path):
    print(f"Cleaning {file_path}...")
    cleaned_content, lines_removed = remove_comments_from_file(file_path)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    return lines_removed

def main():
    base_dir = Path(r'C:\Users\MSI\Desktop\WinCoding\EIC_lab\LLM-QAT-on-gpt2')

    folders = [
        base_dir / 'part3_eval_sp',
        base_dir / 'part3_eval_cpt'
    ]

    summary = []
    total_lines = 0

    for folder in folders:
        py_files = list(folder.glob('*.py'))

        for py_file in py_files:
            lines_removed = clean_file(py_file)
            summary.append((str(py_file), lines_removed))
            total_lines += lines_removed

    print("\n" + "="*80)
    print("SUMMARY OF CLEANED FILES")
    print("="*80)

    for file_path, lines_removed in summary:
        print(f"{file_path}: {lines_removed} lines of comments/docstrings removed")

    print("="*80)
    print(f"TOTAL: {total_lines} lines of comments/docstrings removed from {len(summary)} files")
    print("="*80)

if __name__ == '__main__':
    main()
