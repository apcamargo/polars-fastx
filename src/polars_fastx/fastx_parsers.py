from pathlib import Path
from typing import Iterator, Optional, Union

import polars as pl
from needletail import parse_fastx_file
from polars.io.plugins import register_io_source


def scan_fastx(fastx_file: Union[str, Path]) -> pl.LazyFrame:
    schema = pl.Schema(
        {"identifier": pl.String, "sequence": pl.String, "quality": pl.String}
    )

    def source_generator(
        with_columns: Optional[list[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        if batch_size is None:
            batch_size = 512
        reader = parse_fastx_file(fastx_file)
        while n_rows is None or n_rows > 0:
            if n_rows is not None:
                batch_size = min(batch_size, n_rows)
            rows = []
            for _ in range(batch_size):
                try:
                    record = next(reader)
                    row = [record.id, record.seq, record.qual]
                except StopIteration:
                    n_rows = 0
                    break
                rows.append(row)
            df = pl.from_records(rows, schema=schema, orient="row")
            if n_rows:
                n_rows -= df.height
            if with_columns is not None:
                df = df.select(with_columns)
            if predicate is not None:
                df = df.filter(predicate)
            yield df

    return register_io_source(io_source=source_generator, schema=schema)
