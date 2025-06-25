from pathlib import Path
import sys
from typing import  Optional, Union
from collections import defaultdict

import polars as pl
from needletail import parse_fastx_file

# Register custom expressions for sequence analysis
@pl.api.register_expr_namespace("seq")
class SequenceExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def gc_content(self) -> pl.Expr:
        """Calculate GC content of sequence"""
        return (
            self._expr.str.count_matches("G") + self._expr.str.count_matches("C")
        ) / self._expr.str.len_chars()

    def n_count(self) -> pl.Expr:
        """Count N's in sequence"""
        return self._expr.str.count_matches("N")

    def length(self) -> pl.Expr:
        """Get sequence length"""
        return self._expr.str.len_chars()

    def codon_usage(self) -> pl.Expr:
        """Calculate codon usage frequencies"""
        def _calc_codons(seq: str) -> dict:
            codons = defaultdict(int)
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i : i + 3].upper()
                if "N" not in codon:
                    codons[codon] += 1
            total = sum(codons.values())
            return {k: v / total for k, v in codons.items()} if total > 0 else {}

        return self._expr.map_elements(_calc_codons, return_dtype=pl.Struct)

    def generate_hash(self, length: int = 32) -> pl.Expr:
        """Generate a hash for a sequence"""
        import hashlib

        def _hash(seq: str) -> str:
            return hashlib.md5(seq.encode()).hexdigest()[:length]

        return self._expr.map_elements(_hash, return_dtype=pl.String)

    def calculate_kmer_frequencies(self, k: int = 3) -> pl.Expr:
        """Calculate k-mer frequencies in the sequence"""
        def _calc_kmers(seq: str, k: int) -> dict:
            if not seq or len(seq) < k:
                return {}
            kmers = defaultdict(int)
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k].upper()
                if "N" not in kmer:
                    kmers[kmer] += 1
            total = sum(kmers.values())
            return {k: v / total for k, v in kmers.items()} if total > 0 else {}

        return self._expr.map_elements(
            lambda x: _calc_kmers(x, k), return_dtype=pl.Struct
        )

@pl.api.register_lazyframe_namespace("from_fastx")
def init(input_file: Union[str, Path], batch_size: int = 512) -> pl.LazyFrame:
    """Scan a FASTA/FASTQ file into a lazy polars DataFrame.

    This function extends polars with the ability to lazily read FASTA/FASTQ files.
    It can be used directly as pl.LazyFrame.fastx.scan("sequences.fasta").

    Args:
        path (Union[str, Path]): Path to the FASTA/FASTQ file
        batch_size (int, optional): Number of records to read per batch. Defaults to 512.

    Returns:
        pl.LazyFrame: Lazy DataFrame with columns:
            - header: Sequence headers (str)
            - sequence: Sequences (str) # TODO: maybe somehow move to largeutf8?
            - quality: Quality scores (only for FASTQ)
    """
    def file_has_quality(file: Union[str, Path]) -> bool:
        first_record = next(parse_fastx_file(file))
        return first_record.qual is not None

    has_quality = file_has_quality(input_file)
    if has_quality:
        schema = pl.Schema({"header": pl.String, "sequence": pl.String, "quality": pl.String})
    else:
        schema = pl.Schema({"header": pl.String, "sequence": pl.String})

    def read_chunks():
        reader = parse_fastx_file(input_file)
        while True:
            chunk = []
            for _ in range(batch_size):
                try:
                    record = next(reader)
                    row = [record.id, record.seq]
                    if has_quality:
                        row.append(record.qual)
                    chunk.append(row)
                except StopIteration:
                    if chunk:
                        yield pl.LazyFrame(chunk, schema=schema, orient="row")
                    return
            yield pl.LazyFrame(chunk, schema=schema, orient="row")

    return pl.concat(read_chunks(), how="vertical")



@pl.api.register_dataframe_namespace("from_fastx")
def init(file: Union[str, Path], batch_size: int = 512) -> pl.DataFrame:
    return pl.LazyFrame.from_fastx(file, batch_size).collect()


### example usage
def fasta_stats(
    input_file: str,
    output_file: Optional[str] = None, # if not provided, will print to stdout
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    fields: str = "header,length,gc_content,n_count,hash,codon_usage,kmer_freq",
    kmer_length: int = 3,
) -> None:
    """Calculate sequence statistics using Polars expressions
    
    Args:
        input: Input file or directory
        output: Output path
        min_length: Minimum sequence length to consider
        max_length: Maximum sequence length to consider
        fields: Comma-separated list of fields to include (available: header,sequence,length,gc_content,n_count,hash,codon_usage,kmer_freq)
    """
    output_path = Path(output_file) if output_file else sys.stdout

    # Read sequences into DataFrame
    df = pl.DataFrame.from_fastx(input_file)
    
    # init_height = df.height
    # Apply length filters
    if min_length:
        df = df.filter(pl.col("sequence").seq.length() >= min_length)
    if max_length:
        df = df.filter(pl.col("sequence").seq.length() <= max_length)
    # print(f"Filtered {init_height - df.height} sequences out of {init_height}")

    # Define available fields and their dependencies
    field_options = {
        "length": {"desc": "Sequence length"},
        "gc_content": {"desc": "GC content percentage"},
        "n_count": {"desc": "Count of Ns in sequence"},
        "hash": {"desc": "Sequence hash (MD5)"},
        "codon_usage": {"desc": "Codon usage frequencies"},
        "kmer_freq": {"desc": "K-mer frequencies"},
        "header": {"desc": "Sequence header"},
        "sequence": {"desc": "DNA/RNA sequence"}
    }

    # Parse fields
    selected_fields = ["header"]
    if fields:
        selected_fields = [f.strip().lower() for f in fields.split(",")]
        # Validate fields
        valid_fields = list(field_options.keys())
        invalid_fields = [f for f in selected_fields if f not in valid_fields]
        if invalid_fields:
            print(f"Unknown field(s): {', '.join(invalid_fields)}")
            print(f"Available fields are: {', '.join(valid_fields)}")
        selected_fields = [f for f in selected_fields if f in valid_fields]

    # Build the stats expressions
    stats_expr = []
    # for field in selected_fields: # this doesn't work  :(
    #     stats_expr.append(pl.col("sequence").seq.field(field).alias(field))
    
    if "length" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.length().alias("length"))
    if "gc_content" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.gc_content().alias("gc_content"))
    if "n_count" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.n_count().alias("n_count"))
    if "hash" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.generate_hash().alias("hash")) 
    if "codon_usage" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.codon_usage().alias("codon_usage"))
    if "kmer_freq" in selected_fields:
        stats_expr.append(pl.col("sequence").seq.calculate_kmer_frequencies(kmer_length).alias("kmer_freq"))

    # Apply all the stats expressions
    df = df.with_columns(stats_expr)
    df = df.select(selected_fields)


    # Convert all nested columns to strings
    for col in df.columns:
        if col != "header":  # Keep header as is
            if isinstance(df[col].dtype, pl.Struct) or isinstance(df[col].dtype, pl.List):
                df = df.with_columns(
                    [pl.col(col).cast(pl.Utf8).alias(f"{col}")]
                )

    df.write_csv(output_path, separator="\t")
    print("Successfully wrote file after converting data types")
