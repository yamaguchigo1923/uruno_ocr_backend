from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class OrderSelection(BaseModel):
    maker: str
    code: str
    seibun_flag: str
    mihon_flag: str


class OrderResponse(BaseModel):
    ocr_table: List[List[str]]
    reference_table: List[List[str]]
    selections: List[OrderSelection]
    maker_data: Dict[str, List[List[str]]]
    maker_cds: Dict[str, List[str]]
    flags: List[List[str]]
    ocr_snapshot_url: Optional[str]
    output_spreadsheet_url: Optional[str]
    center_name: str
    center_month: str
    debug_logs: List[str]


class ExportRequest(BaseModel):
    center_id: str
    center_name: str
    center_month: str
    maker_data: Dict[str, List[List[str]]]
    maker_cds: Dict[str, List[str]]
    flags: List[List[str]]


class ExportResponse(BaseModel):
    output_spreadsheet_url: str
    debug_logs: List[str]
