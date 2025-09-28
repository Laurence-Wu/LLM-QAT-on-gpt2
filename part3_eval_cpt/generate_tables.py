from typing import Dict, List
import numpy as np
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

    def tabulate(data, headers='keys', tablefmt='grid', floatfmt='.1f'):
        """Fallback tabulate function when tabulate is not installed"""
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            return data.to_string()
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # Handle list of dictionaries
            if headers == 'keys' and data:
                headers = list(data[0].keys())
            header_str = " | ".join(str(h) for h in headers)
            separator = "-" * len(header_str)
            rows = []
            for row in data:
                row_values = [str(row.get(h, '')) for h in headers]
                rows.append(" | ".join(row_values))
            return f"{header_str}\n{separator}\n" + "\n".join(rows)
        else:
            return str(data)

class ResultTableGenerator:
    def __init__(self, results: Dict):
        self.results = results

    def generate_table_1_zero_shot(self) -> str:
        """
        Generate Table 1: Zero-shot common sense results
        Columns: Method | #Bits | Size(GB) | [Available Tasks] | Avg
        """
        rows = []

        # Get all unique task names from results
        all_tasks = set()
        for config_name, result in self.results.items():
            if 'zero_shot' in result and result['zero_shot']:
                all_tasks.update([k for k in result['zero_shot'].keys() if k != 'Average'])

        # Sort task names for consistent ordering
        task_names = sorted(list(all_tasks))

        for config_name, result in self.results.items():
            if 'zero_shot' not in result or not result['zero_shot']:
                continue

            zero_shot = result['zero_shot']
            row = {
                'Method': config_name,
                '#Bits': result.get('bits', 'N/A'),
                'Size(GB)': result.get('model_size_gb', 0),
            }

            # Add scores for each available task
            for task in task_names:
                row[task] = zero_shot.get(task, 0)

            row['Avg'] = zero_shot.get('Average', 0)
            rows.append(row)

        if not rows:
            return "No zero-shot results available"

        if HAS_PANDAS:
            df = pd.DataFrame(rows)
            df = df.sort_values('Avg', ascending=False)
        else:
            # Simple sort for non-pandas case
            df = sorted(rows, key=lambda x: x.get('Avg', 0), reverse=True)

        table_str = "Table 1: Zero-shot Common Sense Performance (↑)\n"
        table_str += "=" * 100 + "\n"
        table_str += tabulate(df, headers='keys', tablefmt='grid', floatfmt='.1f')

        print("\n" + table_str)

        self._save_to_file(table_str, 'table1_zero_shot.txt')

        return table_str

    def generate_table_2_perplexity(self) -> str:
        """
        Generate Table 2: Perplexity results
        Columns: Method | #Bits | OpenWebText↓ | WikiText2↓
        """
        rows = []

        for config_name, result in self.results.items():
            if 'perplexity' not in result or not result['perplexity']:
                continue

            perplexity = result['perplexity']
            row = {
                'Method': config_name,
                '#Bits': result.get('bits', 'N/A'),
                'OpenWebText↓': perplexity.get('OpenWebText', float('inf')),
                'WikiText2↓': perplexity.get('WikiText2', float('inf'))
            }
            rows.append(row)

        if not rows:
            return "No perplexity results available"

        if HAS_PANDAS:
            df = pd.DataFrame(rows)
            df = df.sort_values('WikiText2↓', ascending=True)
        else:
            # Simple sort for non-pandas case
            df = sorted(rows, key=lambda x: x.get('WikiText2↓', float('inf')))

        table_str = "Table 2: Perplexity Results (↓)\n"
        table_str += "=" * 50 + "\n"
        table_str += tabulate(df, headers='keys', tablefmt='grid', floatfmt='.1f')

        print("\n" + table_str)

        self._save_to_file(table_str, 'table2_perplexity.txt')

        return table_str

    def generate_table_7_few_shot(self) -> str:
        """
        Generate Table 7: Few-shot results
        Columns: Method | MMLU-Hum | MMLU-STEM | MMLU-Social | MMLU-Other | MMLU-Avg | TriviaQA
        """
        rows = []

        for config_name, result in self.results.items():
            if 'few_shot' not in result or not result['few_shot']:
                continue

            few_shot = result['few_shot']
            mmlu = few_shot.get('MMLU', {})

            row = {
                'Method': config_name,
                'MMLU-Hum': mmlu.get('Humanities', 0),
                'MMLU-STEM': mmlu.get('STEM', 0),
                'MMLU-Social': mmlu.get('Social Sciences', 0),
                'MMLU-Other': mmlu.get('Other', 0),
                'MMLU-Avg': mmlu.get('Average', 0),
                'TriviaQA': few_shot.get('TriviaQA', 0)
            }
            rows.append(row)

        if not rows:
            return "No few-shot results available"

        if HAS_PANDAS:
            df = pd.DataFrame(rows)
            df = df.sort_values('MMLU-Avg', ascending=False)
        else:
            # Simple sort for non-pandas case
            df = sorted(rows, key=lambda x: x.get('MMLU-Avg', 0), reverse=True)

        table_str = "Table 7: Few-shot Performance (↑)\n"
        table_str += "=" * 80 + "\n"
        table_str += tabulate(df, headers='keys', tablefmt='grid', floatfmt='.1f')

        print("\n" + table_str)

        self._save_to_file(table_str, 'table7_few_shot.txt')

        return table_str

    def export_to_latex(self) -> Dict[str, str]:
        """Export tables in LaTeX format for paper"""
        latex_tables = {}

        latex_tables['zero_shot'] = self._generate_latex_table_1()
        latex_tables['perplexity'] = self._generate_latex_table_2()

        output_dir = Path('part3_evaluation/results')
        output_dir.mkdir(exist_ok=True, parents=True)

        for name, latex in latex_tables.items():
            with open(output_dir / f'{name}_table.tex', 'w') as f:
                f.write(latex)

        return latex_tables

    def export_to_markdown(self) -> Dict[str, str]:
        """Export tables in Markdown format for README"""
        markdown_tables = {}

        markdown_tables['zero_shot'] = self._generate_markdown_table_1()
        markdown_tables['perplexity'] = self._generate_markdown_table_2()

        output_dir = Path('part3_evaluation/results')
        output_dir.mkdir(exist_ok=True, parents=True)

        combined_markdown = "# LLM-QAT Evaluation Results\n\n"

        for name, md in markdown_tables.items():
            combined_markdown += md + "\n\n"

        with open(output_dir / 'results_tables.md', 'w') as f:
            f.write(combined_markdown)

        print(f"\nMarkdown tables saved to {output_dir / 'results_tables.md'}")

        return markdown_tables

    def _generate_latex_table_1(self) -> str:
        """Generate LaTeX for zero-shot table"""
        latex = r"\begin{table}[h]" + "\n"
        latex += r"\centering" + "\n"
        latex += r"\caption{Zero-shot Common Sense Performance}" + "\n"
        latex += r"\begin{tabular}{l|c|c|cccccccc|c}" + "\n"
        latex += r"\hline" + "\n"
        latex += r"Method & \#Bits & Size(GB) & BoolQ & PIQA & SIQA & HellaSwag & WinoGrande & ARC-e & ARC-c & OBQA & Avg \\" + "\n"
        latex += r"\hline" + "\n"

        for config_name, result in self.results.items():
            if 'zero_shot' not in result:
                continue

            zero_shot = result['zero_shot']
            latex += f"{config_name} & {result.get('bits', 'N/A')} & {result.get('model_size_gb', 0):.1f}"

            # Only include tasks that exist in the results
            available_tasks = [k for k in zero_shot.keys() if k != 'Average']
            for task in sorted(available_tasks):
                latex += f" & {zero_shot.get(task, 0):.1f}"

            latex += f" & {zero_shot.get('Average', 0):.1f} \\\\\n"

        latex += r"\hline" + "\n"
        latex += r"\end{tabular}" + "\n"
        latex += r"\end{table}" + "\n"

        return latex

    def _generate_latex_table_2(self) -> str:
        """Generate LaTeX for perplexity table"""
        latex = r"\begin{table}[h]" + "\n"
        latex += r"\centering" + "\n"
        latex += r"\caption{Perplexity Results}" + "\n"
        latex += r"\begin{tabular}{l|c|cc}" + "\n"
        latex += r"\hline" + "\n"
        latex += r"Method & \#Bits & OpenWebText$\downarrow$ & WikiText2$\downarrow$ \\" + "\n"
        latex += r"\hline" + "\n"

        for config_name, result in self.results.items():
            if 'perplexity' not in result:
                continue

            perplexity = result['perplexity']
            latex += f"{config_name} & {result.get('bits', 'N/A')}"
            latex += f" & {perplexity.get('OpenWebText', float('inf')):.1f}"
            latex += f" & {perplexity.get('WikiText2', float('inf')):.1f} \\\\\n"

        latex += r"\hline" + "\n"
        latex += r"\end{tabular}" + "\n"
        latex += r"\end{table}" + "\n"

        return latex

    def _generate_markdown_table_1(self) -> str:
        """Generate Markdown for zero-shot table"""
        # Get all unique task names from results
        all_tasks = set()
        for config_name, result in self.results.items():
            if 'zero_shot' in result and result['zero_shot']:
                all_tasks.update([k for k in result['zero_shot'].keys() if k != 'Average'])

        task_names = sorted(list(all_tasks))

        md = "## Table 1: Zero-shot Common Sense Performance (↑)\n\n"
        md += "| Method | #Bits | Size(GB) | " + " | ".join(task_names) + " | Avg |\n"
        md += "|--------|-------|----------|" + "|".join(["-" * 7 for _ in task_names]) + "|-----|\n"

        for config_name, result in self.results.items():
            if 'zero_shot' not in result or not result['zero_shot']:
                continue

            zero_shot = result['zero_shot']
            md += f"| {config_name} | {result.get('bits', 'N/A')} | {result.get('model_size_gb', 0):.1f}"

            # Only include tasks that exist in the results
            available_tasks = [k for k in zero_shot.keys() if k != 'Average']
            for task in sorted(available_tasks):
                md += f" | {zero_shot.get(task, 0):.1f}"

            md += f" | **{zero_shot.get('Average', 0):.1f}** |\n"

        return md

    def _generate_markdown_table_2(self) -> str:
        """Generate Markdown for perplexity table"""
        md = "## Table 2: Perplexity Results (↓)\n\n"
        md += "| Method | #Bits | OpenWebText↓ | WikiText2↓ |\n"
        md += "|--------|-------|-----|------------|\n"

        for config_name, result in self.results.items():
            if 'perplexity' not in result or not result['perplexity']:
                continue

            perplexity = result['perplexity']
            md += f"| {config_name} | {result.get('bits', 'N/A')}"
            md += f" | {perplexity.get('OpenWebText', float('inf')):.1f}"
            md += f" | {perplexity.get('WikiText2', float('inf')):.1f} |\n"

        return md

    def _save_to_file(self, content: str, filename: str):
        """Save content to file"""
        output_dir = Path('part3_evaluation/results')
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(output_dir / filename, 'w') as f:
            f.write(content)