"""
Dataset Exporters for Data Smith.
Export generated datasets to JSON, JSONL, CSV, and Parquet formats.
"""

import json
import csv
from io import StringIO, BytesIO
from typing import List, Dict, Any, Literal, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime

# Parquet support (optional)
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class ExportError(Exception):
    """Exception raised during export."""
    pass


class DatasetExporter(ABC):
    """Abstract base class for dataset exporters."""
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Get the format name."""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Get the file extension."""
        pass
    
    @property
    @abstractmethod
    def mime_type(self) -> str:
        """Get the MIME type."""
        pass
    
    @abstractmethod
    def export(self, data: List[Dict[str, Any]]) -> bytes:
        """
        Export data to the target format.
        
        Args:
            data: List of dataset items
            
        Returns:
            Exported data as bytes
        """
        pass
    
    def export_to_file(self, data: List[Dict[str, Any]], filepath: str):
        """Export data directly to a file."""
        content = self.export(data)
        with open(filepath, 'wb') as f:
            f.write(content)


class JSONExporter(DatasetExporter):
    """Export dataset as JSON array."""
    
    def __init__(self, indent: int = 2, ensure_ascii: bool = False):
        self.indent = indent
        self.ensure_ascii = ensure_ascii
    
    @property
    def format_name(self) -> str:
        return "json"
    
    @property
    def file_extension(self) -> str:
        return ".json"
    
    @property
    def mime_type(self) -> str:
        return "application/json"
    
    def export(self, data: List[Dict[str, Any]]) -> bytes:
        """Export as formatted JSON array."""
        return json.dumps(
            data, 
            indent=self.indent, 
            ensure_ascii=self.ensure_ascii
        ).encode('utf-8')


class JSONLExporter(DatasetExporter):
    """Export dataset as JSON Lines (one JSON object per line)."""
    
    def __init__(self, ensure_ascii: bool = False):
        self.ensure_ascii = ensure_ascii
    
    @property
    def format_name(self) -> str:
        return "jsonl"
    
    @property
    def file_extension(self) -> str:
        return ".jsonl"
    
    @property
    def mime_type(self) -> str:
        return "application/x-ndjson"
    
    def export(self, data: List[Dict[str, Any]]) -> bytes:
        """Export as JSON Lines."""
        lines = []
        for item in data:
            lines.append(json.dumps(item, ensure_ascii=self.ensure_ascii))
        return '\n'.join(lines).encode('utf-8')


class CSVExporter(DatasetExporter):
    """Export dataset as CSV."""
    
    def __init__(self, delimiter: str = ',', flatten_nested: bool = True):
        self.delimiter = delimiter
        self.flatten_nested = flatten_nested
    
    @property
    def format_name(self) -> str:
        return "csv"
    
    @property
    def file_extension(self) -> str:
        return ".csv"
    
    @property
    def mime_type(self) -> str:
        return "text/csv"
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionaries."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                # Convert list to JSON string
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def export(self, data: List[Dict[str, Any]]) -> bytes:
        """Export as CSV."""
        if not data:
            return b''
        
        # Flatten if needed
        if self.flatten_nested:
            data = [self._flatten_dict(item) for item in data]
        
        # Get all unique keys for headers
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        headers = sorted(all_keys)
        
        # Write CSV
        output = StringIO()
        writer = csv.DictWriter(
            output, 
            fieldnames=headers, 
            delimiter=self.delimiter,
            extrasaction='ignore'
        )
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue().encode('utf-8')


class ParquetExporter(DatasetExporter):
    """Export dataset as Parquet (requires pyarrow)."""
    
    def __init__(self, compression: str = 'snappy'):
        if not PARQUET_AVAILABLE:
            raise ExportError("Parquet export requires pyarrow. Install with: pip install pyarrow")
        self.compression = compression
    
    @property
    def format_name(self) -> str:
        return "parquet"
    
    @property
    def file_extension(self) -> str:
        return ".parquet"
    
    @property
    def mime_type(self) -> str:
        return "application/octet-stream"
    
    def export(self, data: List[Dict[str, Any]]) -> bytes:
        """Export as Parquet."""
        if not data:
            return b''
        
        # Convert nested structures to JSON strings
        processed_data = []
        for item in data:
            processed_item = {}
            for k, v in item.items():
                if isinstance(v, (dict, list)):
                    processed_item[k] = json.dumps(v)
                else:
                    processed_item[k] = v
            processed_data.append(processed_item)
        
        # Create table from list of dicts
        table = pa.Table.from_pylist(processed_data)
        
        # Write to bytes
        buffer = BytesIO()
        pq.write_table(table, buffer, compression=self.compression)
        return buffer.getvalue()


class ExporterFactory:
    """Factory for creating dataset exporters."""
    
    _exporters = {
        'json': JSONExporter,
        'jsonl': JSONLExporter,
        'csv': CSVExporter,
    }
    
    if PARQUET_AVAILABLE:
        _exporters['parquet'] = ParquetExporter
    
    @classmethod
    def get_exporter(
        cls, 
        format_type: Literal['json', 'jsonl', 'csv', 'parquet'],
        **kwargs
    ) -> DatasetExporter:
        """
        Get an exporter for the specified format.
        
        Args:
            format_type: Export format
            **kwargs: Additional arguments for the exporter
            
        Returns:
            DatasetExporter instance
        """
        if format_type not in cls._exporters:
            raise ExportError(f"Unknown export format: {format_type}")
        
        return cls._exporters[format_type](**kwargs)
    
    @classmethod
    def available_formats(cls) -> List[str]:
        """Get list of available export formats."""
        return list(cls._exporters.keys())
    
    @classmethod
    def format_info(cls) -> List[Dict[str, str]]:
        """Get info about available formats."""
        info = []
        for fmt in cls._exporters.keys():
            exporter = cls._exporters[fmt]()
            info.append({
                "format": exporter.format_name,
                "extension": exporter.file_extension,
                "mime_type": exporter.mime_type
            })
        return info


# Convenience function
def export_dataset(
    data: List[Dict[str, Any]],
    format_type: Literal['json', 'jsonl', 'csv', 'parquet'] = 'json'
) -> bytes:
    """
    Quick function to export a dataset.
    
    Args:
        data: Dataset items
        format_type: Export format
        
    Returns:
        Exported data as bytes
    """
    exporter = ExporterFactory.get_exporter(format_type)
    return exporter.export(data)
