import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import base64
from io import BytesIO
import importlib.util

# Initialize flags for optional dependencies
HAS_FITZ = False
HAS_LANGDETECT = False
HAS_PIL = False

# Import optional dependencies
try:
    import fitz  # PyMuPDF for PDF comparison
    HAS_FITZ = True
except ImportError:
    logging.warning("PyMuPDF (fitz) not found. PDF comparison features will be disabled.")

try:
    from langdetect import detect_langs
    HAS_LANGDETECT = True
except ImportError:
    logging.warning("langdetect not found. Language detection features will be disabled.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    logging.warning("Pillow not found. Image analysis features will be disabled.")

# Required dependencies
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich import print as rprint
from config import Config

# --- Setup Rich Console ---
console = Console()

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

class DoclingInspector:
    """
    A class to inspect and analyze Docling JSON output and compare with original PDF
    """

class PersianDocAnalyzer(DoclingInspector):
    """
    Specialized analyzer for Persian documents with enhanced quality metrics
    """
    def __init__(self, json_path: Path, pdf_path: Optional[Path] = None):
        self.json_path = json_path
        self.pdf_path = pdf_path
        self.console = Console()
        self.data = None
        self.pdf_doc = None
        self.text_quality = PersianTextQuality()
        
        self.load_data()
        if pdf_path and HAS_FITZ:
            self.load_pdf()

    def load_data(self):
        """Load and validate the Docling JSON output"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found at: {self.json_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def load_pdf(self):
        """Load the original PDF for comparison"""
        if not HAS_FITZ:
            self.console.print("[yellow]PDF comparison is not available. Please install PyMuPDF package.[/yellow]")
            return

        if self.pdf_path and self.pdf_path.exists():
            try:
                self.pdf_doc = fitz.open(self.pdf_path)
            except Exception as e:
                self.console.print(f"[red]Error loading PDF: {str(e)}[/red]")
        else:
            self.console.print("[yellow]Warning: PDF file not found or not specified[/yellow]")

    def get_element_stats(self) -> Dict[str, Dict[int, int]]:
        """Collect detailed statistics about elements per page"""
        stats = defaultdict(lambda: defaultdict(int))
        
        # Process text elements
        texts = self.data.get('texts', [])
        if texts:
            stats['Text'][1] = len(texts)  # Assuming all texts are on page 1 for now
            
        # Process tables
        tables = self.data.get('tables', [])
        if tables:
            stats['Table'][1] = len(tables)
            
        # Process pictures
        pictures = self.data.get('pictures', [])
        if pictures:
            stats['Image'][1] = len(pictures)
            
        # Process groups (structural elements)
        groups = self.data.get('groups', [])
        if groups:
            for group in groups:
                group_label = group.get('label')
                if group_label:
                    stats[group_label.capitalize()][1] += 1
                    
        # Process key-value items
        kv_items = self.data.get('key_value_items', [])
        if kv_items:
            stats['KeyValue'][1] = len(kv_items)
            
        # Process form items
        form_items = self.data.get('form_items', [])
        if form_items:
            stats['Form'][1] = len(form_items)
                        
        return stats

    def print_element_distribution(self):
        """Display element distribution across pages in a table format"""
        stats = self.get_element_stats()
        
        if not stats:
            self.console.print("[yellow]No structured elements found in the document[/yellow]")
            return
            
        table = Table(title="Element Distribution Across Pages")
        table.add_column("Element Type", style="cyan")
        table.add_column("Total Count", style="magenta")
        table.add_column("Page Distribution", style="green")
        table.add_column("Sample Content", style="white")
        
        for el_type, page_counts in stats.items():
            total = sum(page_counts.values())
            distribution = [f"P{page}: {count}" for page, count in sorted(page_counts.items())]
            
            # Get a sample content for this element type
            sample = self._get_sample_content(el_type)
            
            table.add_row(
                el_type,
                str(total),
                ", ".join(distribution),
                sample if sample else "-"
            )
        
        self.console.print(table)

    def _get_sample_content(self, element_type: str) -> str:
        """Get a sample content for the given element type"""
        pages = self.data.get('pages', [])
        
        for page in pages:
            if not isinstance(page, dict):
                continue
                
            for element in page.get('elements', []):
                if isinstance(element, dict) and element.get('type') == element_type:
                    if element_type == 'Table':
                        rows = len(element.get('rows', []))
                        cols = len(element.get('rows', [[]])[0]) if element.get('rows') else 0
                        return f"{rows}x{cols} table"
                    elif 'text' in element:
                        text = element['text'].replace('\n', ' ').strip()
                        return f"{text[:50]}..." if len(text) > 50 else text
                    
        return ""

    def analyze_text_quality(self, page_num: int = 1):
        """Analyze text quality by comparing with original PDF if available"""
        pages = self.data.get('pages', [])
        target_page = None
        
        for page in pages:
            if page.get('page_num') == page_num:
                target_page = page
                break
                
        if not target_page:
            self.console.print(f"[red]Page {page_num} not found in JSON[/red]")
            return

        # Extract text from Docling output
        docling_text = []
        for element in target_page.get('elements', []):
            if isinstance(element, dict) and element.get('text'):
                docling_text.append(element['text'])

        # Compare with PDF if available
        if self.pdf_doc and 0 <= page_num - 1 < len(self.pdf_doc):
            pdf_page = self.pdf_doc[page_num - 1]
            pdf_text = pdf_page.get_text()
            
            self.console.print("\n[bold cyan]Text Comparison Analysis[/bold cyan]")
            self.console.print("\n[bold green]Docling Extracted Text:[/bold green]")
            self.console.print("\n".join(docling_text))
            self.console.print("\n[bold yellow]Original PDF Text:[/bold yellow]")
            self.console.print(pdf_text)
        else:
            self.console.print("\n[bold green]Docling Extracted Text:[/bold green]")
            self.console.print("\n".join(docling_text))

    def analyze_tables(self):
        """Analyze table structures and their content"""
        pages = self.data.get('pages', [])
        table_count = 0
        
        table = Table(title="Table Analysis")
        table.add_column("Page", style="cyan")
        table.add_column("Rows", style="magenta")
        table.add_column("Columns", style="green")
        table.add_column("Cell Count", style="yellow")
        table.add_column("Headers", style="blue")
        
        for page in pages:
            # Skip if page is not a dictionary
            if not isinstance(page, dict):
                continue
                
            for element in page.get('elements', []):
                if isinstance(element, dict) and element.get('type') == 'Table':
                    table_count += 1
                    rows = element.get('rows', [])
                    row_count = len(rows)
                    col_count = len(rows[0]) if rows else 0
                    cells = row_count * col_count
                    
                    # Try to detect headers
                    headers = "Yes" if self._detect_table_headers(rows) else "No"
                    
                    table.add_row(
                        str(page.get('page_num')),
                        str(row_count),
                        str(col_count),
                        str(cells),
                        headers
                    )
        
        if table_count > 0:
            self.console.print(table)
        else:
            self.console.print("[yellow]No tables found in the document[/yellow]")

    def _detect_table_headers(self, rows: List[List[str]]) -> bool:
        """Helper method to detect if a table has headers"""
        if not rows or len(rows) < 2:
            return False
        
        # Heuristics for header detection
        first_row = rows[0]
        second_row = rows[1]
        
        # Check if first row is shorter (merged cells often indicate headers)
        if len(first_row) != len(second_row):
            return True
            
        # Check if first row has different formatting (bold, different font, etc.)
        # This would require format information in the JSON
        
        return False

    def analyze_language_distribution(self):
        """Analyze language distribution in the document"""
        if not HAS_LANGDETECT:
            self.console.print("[yellow]Language detection is not available. Please install langdetect package.[/yellow]")
            return

        pages = self.data.get('pages', [])
        language_stats = defaultdict(int)
        total_text_blocks = 0
        
        table = Table(title="Language Analysis")
        table.add_column("Language", style="cyan")
        table.add_column("Confidence", style="magenta")
        table.add_column("Text Sample", style="green")
        
        for page in pages:
            for element in page.get('elements', []):
                if isinstance(element, dict) and element.get('text'):
                    total_text_blocks += 1
                    text = element['text']
                    try:
                        langs = detect_langs(text)
                        for lang in langs:
                            language_stats[lang.lang] += lang.prob
                            if len(text) > 50:  # Only show longer text samples
                                table.add_row(
                                    lang.lang,
                                    f"{lang.prob:.2f}",
                                    text[:100] + "..."
                                )
                    except Exception:
                        continue
        
        # Normalize and display language distribution
        if language_stats:
            self.console.print("\n[bold]Language Distribution Analysis[/bold]")
            self.console.print(table)
            
            # Overall statistics
            total_prob = sum(language_stats.values())
            if total_prob > 0:
                self.console.print("\n[bold]Overall Language Distribution:[/bold]")
                for lang, prob in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
                    percentage = (prob / total_prob) * 100
                    self.console.print(f"{lang}: {percentage:.1f}%")

    def _add_structure_node(self, node: dict, tree_node: Tree):
        """Helper method to recursively add structure nodes to the tree"""
        if not isinstance(node, dict):
            return
            
        # Add children
        children = node.get('children', [])
        for child in children:
            if isinstance(child, dict):
                label = child.get('label', 'unknown')
                name = child.get('name', '')
                content_layer = child.get('content_layer', '')
                child_node = tree_node.add(f"[cyan]{label}[/cyan] ({content_layer})")
                self._add_structure_node(child, child_node)

    def analyze_document_structure(self):
        """Analyze and visualize the document's structural hierarchy"""
        doc_tree = Tree("[bold cyan]Document Structure[/bold cyan]")
        
        # Add basic document info
        doc_name = self.data.get('name', 'Unnamed Document')
        doc_version = self.data.get('version', 'Unknown')
        doc_tree.add(f"[bold white]Name: {doc_name}[/bold white]")
        doc_tree.add(f"[bold white]Version: {doc_version}[/bold white]")
        
        # Add body structure
        body = self.data.get('body', {})
        if body:
            body_node = doc_tree.add("[bold magenta]Body Content[/bold magenta]")
            self._add_structure_node(body, body_node)
            
        # Add furniture structure
        furniture = self.data.get('furniture', {})
        if furniture:
            furniture_node = doc_tree.add("[bold yellow]Document Furniture[/bold yellow]")
            self._add_structure_node(furniture, furniture_node)
            
        # Add groups
        groups = self.data.get('groups', [])
        if groups:
            groups_node = doc_tree.add("[bold green]Content Groups[/bold green]")
            for group in groups:
                if isinstance(group, dict):
                    label = group.get('label', 'Unnamed Group')
                    groups_node.add(f"[green]{label}[/green]")
        
        # Add content statistics
        stats_node = doc_tree.add("[bold blue]Content Statistics[/bold blue]")
        
        texts = self.data.get('texts', [])
        if texts:
            stats_node.add(f"[blue]Text Elements: {len(texts)}[/blue]")
            
        tables = self.data.get('tables', [])
        if tables:
            stats_node.add(f"[cyan]Tables: {len(tables)}[/cyan]")
            
        pictures = self.data.get('pictures', [])
        if pictures:
            stats_node.add(f"[magenta]Pictures: {len(pictures)}[/magenta]")
            
        key_values = self.data.get('key_value_items', [])
        if key_values:
            stats_node.add(f"[yellow]Key-Value Pairs: {len(key_values)}[/yellow]")
            
        form_items = self.data.get('form_items', [])
        if form_items:
            stats_node.add(f"[green]Form Items: {len(form_items)}[/green]")
                
            page_node = doc_tree.add(f"[magenta]Page {page_num}[/magenta]")
            
            # Group elements by type
            elements_by_type = defaultdict(list)
            for element in elements:
                if isinstance(element, dict):
                    el_type = element.get('type', 'Unknown')
                    elements_by_type[el_type].append(element)
            
            # Add element types to the tree
            for el_type, el_list in elements_by_type.items():
                type_node = page_node.add(f"[green]{el_type}s ({len(el_list)})[/green]")
                
                # Add some details for each element type
                if el_type == 'Table':
                    for table in el_list:
                        rows = len(table.get('rows', []))
                        cols = len(table.get('rows', [[]])[0]) if table.get('rows') else 0
                        type_node.add(f"[yellow]{rows}x{cols} Table[/yellow]")
                elif el_type == 'Title':
                    for title in el_list:
                        if title.get('text'):
                            type_node.add(f"[blue]{title['text'][:50]}...[/blue]")
                elif el_type in ['Paragraph', 'ListItem']:
                    for item in el_list[:3]:  # Show only first 3 items to avoid clutter
                        if item.get('text'):
                            preview = item['text'][:50].replace('\n', ' ').strip()
                            type_node.add(f"[white]{preview}...[/white]")
                    if len(el_list) > 3:
                        type_node.add(f"[dim]... and {len(el_list) - 3} more[/dim]")
            
            if not elements_by_type:
                page_node.add("[yellow]No structured elements found[/yellow]")
        
        self.console.print(doc_tree)

    def analyze_image_content(self):
        """Analyze images found in the document"""
        if not HAS_PIL:
            self.console.print("[yellow]Image analysis is not available. Please install Pillow package.[/yellow]")
            return

        pages = self.data.get('pages', [])
        image_count = 0
        
        table = Table(title="Image Analysis")
        table.add_column("Page", style="cyan")
        table.add_column("Size", style="magenta")
        table.add_column("Format", style="green")
        table.add_column("Location", style="yellow")
        
        for page in pages:
            for element in page.get('elements', []):
                if isinstance(element, dict) and element.get('type') == 'Image':
                    image_count += 1
                    
                    # Try to get image data if available
                    image_data = element.get('data', '')
                    if image_data and image_data.startswith('data:image'):
                        try:
                            # Extract actual image data from base64
                            img_format = image_data.split(';')[0].split('/')[-1]
                            img_bytes = base64.b64decode(image_data.split(',')[1])
                            img = Image.open(BytesIO(img_bytes))
                            
                            table.add_row(
                                str(page.get('page_num')),
                                f"{img.width}x{img.height}",
                                img_format,
                                f"({element.get('x', '?')}, {element.get('y', '?')})"
                            )
                        except Exception as e:
                            self.console.print(f"[red]Error analyzing image: {str(e)}[/red]")
                            continue
        
        if image_count > 0:
            self.console.print("\n[bold]Image Content Analysis[/bold]")
            self.console.print(table)
        else:
            self.console.print("[yellow]No images found in the document[/yellow]")

    def analyze_formatting(self):
        """Analyze text formatting and styles"""
        pages = self.data.get('pages', [])
        styles = defaultdict(int)
        fonts = defaultdict(int)
        sizes = defaultdict(int)
        
        for page in pages:
            for element in page.get('elements', []):
                if isinstance(element, dict) and element.get('style'):
                    style = element.get('style', {})
                    
                    # Count font usage
                    if 'font' in style:
                        fonts[style['font']] += 1
                    
                    # Count font sizes
                    if 'fontSize' in style:
                        sizes[style['fontSize']] += 1
                    
                    # Count other style attributes
                    for key, value in style.items():
                        if key not in ['font', 'fontSize']:
                            styles[f"{key}: {value}"] += 1
        
        # Display font statistics
        if fonts:
            font_table = Table(title="Font Usage Analysis")
            font_table.add_column("Font Family", style="cyan")
            font_table.add_column("Usage Count", style="magenta")
            
            for font, count in sorted(fonts.items(), key=lambda x: x[1], reverse=True):
                font_table.add_row(font, str(count))
            
            self.console.print(font_table)
        
        # Display font size distribution
        if sizes:
            size_table = Table(title="Font Size Distribution")
            size_table.add_column("Size", style="cyan")
            size_table.add_column("Usage Count", style="magenta")
            
            for size, count in sorted(sizes.items(), key=lambda x: float(x[0])):
                size_table.add_row(str(size), str(count))
            
            self.console.print(size_table)
        
        # Display other style attributes
        if styles:
            style_table = Table(title="Style Attributes")
            style_table.add_column("Style Property", style="cyan")
            style_table.add_column("Usage Count", style="magenta")
            
            for style, count in sorted(styles.items(), key=lambda x: x[1], reverse=True):
                style_table.add_row(style, str(count))
            
            self.console.print(style_table)

def find_corresponding_pdf(json_path: Path) -> Optional[Path]:
    """Find the corresponding PDF file for a given JSON output"""
    pdf_name = json_path.stem + ".pdf"
    possible_locations = [
        Config.PDF_DIR / pdf_name,
        Config.PDF_DIR.parent / pdf_name
    ]
    
    for loc in possible_locations:
        if loc.exists():
            return loc
    return None

# Add Persian-specific analysis features to the analyzer
class PersianTextQuality:
    """Helper class for Persian text quality analysis"""
    # Persian characters and punctuation marks
    PERSIAN_CHARS = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
    PERSIAN_NUMBERS = set('۰۱۲۳۴۵۶۷۸۹')
    PERSIAN_PUNCTUATION = set('،؛؟»«')
    ARABIC_CHARS = set('إأآةۀكي')
    
    @staticmethod
    def is_persian_text(text: str) -> bool:
        """Check if text is primarily Persian"""
        if not text:
            return False
        text_chars = set(text)
        persian_count = len(text_chars & PersianTextQuality.PERSIAN_CHARS)
        arabic_count = len(text_chars & PersianTextQuality.ARABIC_CHARS)
        return persian_count > arabic_count
    
    @staticmethod
    def get_persian_char_ratio(text: str) -> float:
        """Calculate ratio of Persian characters in text"""
        if not text:
            return 0.0
        persian_chars = sum(1 for c in text if c in PersianTextQuality.PERSIAN_CHARS)
        total_chars = len(text)
        return persian_chars / total_chars if total_chars > 0 else 0.0
    
    @staticmethod
    def has_proper_numbers(text: str) -> bool:
        """Check if text uses Persian numbers consistently"""
        has_persian = any(n in text for n in PersianTextQuality.PERSIAN_NUMBERS)
        has_english = any(n in text for n in '0123456789')
        return has_persian and not has_english

    def analyze_persian_text_quality(self):
        """Analyze the quality of Persian text in the document"""
        texts = self.data.get('texts', [])
        total_texts = 0
        persian_texts = 0
        proper_number_texts = 0
        
        quality_table = Table(title="Persian Text Quality Analysis")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="magenta")
        quality_table.add_column("Status", style="green")
        
        # Process text elements
        for text_element in texts:
            if isinstance(text_element, dict):
                text = text_element.get('text', '')
                if text:
                    total_texts += 1
                    
                    # Check Persian text quality
                    if PersianTextQuality.is_persian_text(text):
                        persian_texts += 1
                    
                    # Check number format
                    if PersianTextQuality.has_proper_numbers(text):
                        proper_number_texts += 1
        
        if total_texts > 0:
            persian_ratio = (persian_texts / total_texts) * 100
            number_ratio = (proper_number_texts / total_texts) * 100
            
            quality_table.add_row(
                "Persian Text Ratio",
                f"{persian_ratio:.1f}%",
                "✅" if persian_ratio > 90 else "⚠️"
            )
            quality_table.add_row(
                "Proper Number Format",
                f"{number_ratio:.1f}%",
                "✅" if number_ratio > 90 else "⚠️"
            )
            
            self.console.print(quality_table)
        else:
            self.console.print("[yellow]No text content found for analysis[/yellow]")

    def analyze_persian_structure(self):
        """Analyze the structural elements specific to Persian documents"""
        pages = self.data.get('pages', [])
        structure_table = Table(title="Persian Document Structure Analysis")
        structure_table.add_column("Element Type", style="cyan")
        structure_table.add_column("Count", style="magenta")
        structure_table.add_column("Quality", style="green")
        
        element_counts = defaultdict(int)
        
        for page in pages:
            if not isinstance(page, dict):
                continue
                
            for element in page.get('elements', []):
                if isinstance(element, dict):
                    element_type = element.get('type', 'Unknown')
                    element_counts[element_type] += 1
        
        # Analyze structure quality
        has_title = element_counts.get('Title', 0) > 0
        has_headings = element_counts.get('Heading', 0) > 0
        has_paragraphs = element_counts.get('Paragraph', 0) > 0
        
        structure_table.add_row(
            "Document Title",
            str(element_counts.get('Title', 0)),
            "✅" if has_title else "❌"
        )
        structure_table.add_row(
            "Headings",
            str(element_counts.get('Heading', 0)),
            "✅" if has_headings else "⚠️"
        )
        structure_table.add_row(
            "Paragraphs",
            str(element_counts.get('Paragraph', 0)),
            "✅" if has_paragraphs else "⚠️"
        )
        
        self.console.print(structure_table)

    def generate_quality_report(self):
        """Generate an overall quality report for the Persian document"""
        report_table = Table(title="Overall Document Quality Report")
        report_table.add_column("Category", style="cyan")
        report_table.add_column("Status", style="magenta")
        report_table.add_column("Notes", style="green")
        
        # Check document structure
        pages = self.data.get('pages', [])
        total_elements = sum(
            len(page.get('elements', [])) 
            for page in pages 
            if isinstance(page, dict)
        )
        
        # Basic structure check
        report_table.add_row(
            "Basic Structure",
            "✅" if total_elements > 0 else "❌",
            f"Found {total_elements} elements"
        )
        
        # Table check
        tables = sum(
            1 for page in pages 
            if isinstance(page, dict)
            for element in page.get('elements', [])
            if isinstance(element, dict) and element.get('type') == 'Table'
        )
        report_table.add_row(
            "Tables",
            "✅" if tables > 0 else "ℹ️",
            f"Found {tables} tables"
        )
        
        # Image check
        images = sum(
            1 for page in pages 
            if isinstance(page, dict)
            for element in page.get('elements', [])
            if isinstance(element, dict) and element.get('type') == 'Image'
        )
        report_table.add_row(
            "Images",
            "✅" if images > 0 else "ℹ️",
            f"Found {images} images"
        )
        
        self.console.print(report_table)

if __name__ == "__main__":
    output_files = list(Config.DOCLING_OUTPUT_DIR.glob("*.json"))
    
    if not output_files:
        console.print("[red]No JSON files found in '{Config.DOCLING_OUTPUT_DIR}'.[/red]")
        console.print("[red]Please run 'main_harvester_docling.py' first.[/red]")
    else:
        for json_file in output_files:
            console.rule(f"[bold cyan]Analyzing {json_file.name}[/bold cyan]")
            
            # Try to find corresponding PDF
            pdf_path = find_corresponding_pdf(json_file)
            if pdf_path:
                console.print(f"[green]Found corresponding PDF: {pdf_path.name}[/green]")
            else:
                console.print("[yellow]No corresponding PDF found for comparison[/yellow]")
            
            # Create specialized Persian document analyzer
            analyzer = PersianDocAnalyzer(json_file, pdf_path)
            
            # Run standard analysis
            console.rule("[bold cyan]Element Distribution[/bold cyan]")
            analyzer.print_element_distribution()
            
            console.rule("[bold cyan]Document Structure Analysis[/bold cyan]")
            analyzer.analyze_document_structure()
            
            console.rule("[bold cyan]Table Analysis[/bold cyan]")
            analyzer.analyze_tables()
            
            # Run Persian-specific analysis
            console.rule("[bold cyan]Persian Text Quality Analysis[/bold cyan]")
            analyzer.analyze_persian_text_quality()
            
            console.rule("[bold cyan]Persian Structure Quality[/bold cyan]")
            analyzer.analyze_persian_structure()
            
            # Run general analysis with Persian context
            console.rule("[bold cyan]Language Analysis[/bold cyan]")
            analyzer.analyze_language_distribution()
            
            console.rule("[bold cyan]Formatting Analysis[/bold cyan]")
            analyzer.analyze_formatting()
            
            console.rule("[bold cyan]Image Analysis[/bold cyan]")
            analyzer.analyze_image_content()
            
            # Generate quality report
            console.rule("[bold cyan]Persian Document Quality Report[/bold cyan]")
            analyzer.generate_quality_report()
            
            console.print("\n")