from __future__ import annotations

import random
import time
import os
from typing import Dict, List, Optional, Sequence, Tuple, Callable

from googleapiclient.errors import HttpError

from config.settings import Settings
from services.google_clients import GoogleClients
from utils.logger import get_logger

logger = get_logger("services.sheet_generator")

MakerData = Dict[str, List[List[str]]]
MakerCodes = Dict[str, List[str]]
FlagsList = List[List[str]]  # [依頼先?, メーカー, 商品CD, 成分表フラグ, 見本フラグ] 形式を想定


class SheetGenerator:
    def __init__(self, settings: Settings, clients: GoogleClients) -> None:
        self.settings = settings
        self.clients = clients
        self.debug_events: List[str] = []
        # Optional external log callback (callable taking a single str). When
        # set by the caller, `_dbg` will invoke it immediately so callers can
        # stream logs (eg. SSE).
        self.log_fn: Optional[Callable[[str], None]] = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _status_of_http_error(self, error: HttpError) -> Optional[int]:
        return getattr(error, "status_code", None) or getattr(getattr(error, "resp", None), "status", None)

    def _execute_with_backoff(self, api_call, label: str, max_retries: int = 8, base_delay: float = 0.6):
        retry_statuses = {408, 429, 500, 502, 503, 504}
        delay = base_delay
        for attempt in range(max_retries):
            try:
                result = api_call.execute()
                self._dbg(f"[OK][{label}] try={attempt}")
                return result
            except HttpError as error:
                status = self._status_of_http_error(error)
                if status in retry_statuses and attempt < max_retries - 1:
                    self._dbg(f"[RETRY][{label}] HttpError {status}; sleep {delay:.2f}s")
                    time.sleep(delay + random.uniform(0, delay * 0.3))
                    delay = min(delay * 2, 30)
                    continue
                self._dbg(f"[ERROR][{label}] HttpError {status}: {error}")
                raise
            except Exception as exc:
                if attempt < max_retries - 1:
                    self._dbg(f"[RETRY][{label}] {exc.__class__.__name__}; sleep {delay:.2f}s")
                    time.sleep(delay + random.uniform(0, delay * 0.3))
                    delay = min(delay * 2, 30)
                    continue
                self._dbg(f"[ERROR][{label}] {exc}")
                raise
        raise RuntimeError(f"API call {label} failed after retries")

    def _dbg(self, message: str) -> None:
        logger.debug(message)
        # If an external log_fn is configured (usually wired to the
        # pipeline's _log), we call it so callers receive realtime events.
        # The pipeline's _log will add a timestamp, so forward the raw
        # message in that case to avoid double timestamps.
        if getattr(self, "log_fn", None):
            try:
                # Keep this as debug so server console (INFO) doesn't show stream internals
                logger.debug("sheet_generator._dbg: invoking log_fn")
                self.log_fn(message)
            except Exception:
                logger.exception("sheet_generator.log_fn failed")
        else:
            # No external callback — append a timestamped message to
            # debug_events so returned logs include emission time.
            from datetime import datetime

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ts_msg = f"[{ts}] {message}"
            self.debug_events.append(ts_msg)

    # ------------------------------------------------------------------
    # Sheets helpers
    # ------------------------------------------------------------------
    def clear_sheet(self, spreadsheet_id: str, range_a1: str) -> None:
        self._execute_with_backoff(
            self.clients.sheets.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id, range=range_a1
            ),
            label="values.clear",
        )

    def update_values(self, spreadsheet_id: str, range_a1: str, values: List[List[str]], label: str) -> None:
        self._execute_with_backoff(
            self.clients.sheets.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_a1,
                valueInputOption="USER_ENTERED",
                body={"values": values},
            ),
            label=label,
        )

    def batch_update_values(self, spreadsheet_id: str, data: List[Dict]) -> None:
        self._execute_with_backoff(
            self.clients.sheets.spreadsheets().values().batchUpdate(
                spreadsheetId=spreadsheet_id, body={"valueInputOption": "USER_ENTERED", "data": data}
            ),
            label="values.batchUpdate",
        )

    def batch_get_values(self, spreadsheet_id: str, ranges: Sequence[str], label: str):
        return self._execute_with_backoff(
            self.clients.sheets.spreadsheets().values().batchGet(
                spreadsheetId=spreadsheet_id,
                ranges=list(ranges),
                valueRenderOption="FORMATTED_VALUE",
            ),
            label=label,
        )

    # ------------------------------------------------------------------
    # Domain specific helpers
    # ------------------------------------------------------------------
    def create_folder(self, title: str, parent_folder_id: Optional[str] = None) -> str:
        """Drive 上に新規フォルダを作成しフォルダIDを返す。"""
        body = {"name": title, "mimeType": "application/vnd.google-apps.folder"}
        if parent_folder_id:
            body["parents"] = [parent_folder_id]
        res = self._execute_with_backoff(
            self.clients.drive.files().create(body=body, supportsAllDrives=True, fields="id"),
            label="files.create[folder]",
        )
        folder_id = res.get("id")
        self._dbg(f"[FOLDER] created id={folder_id} title={title}")
        return folder_id  # type: ignore[return-value]

    def create_basic_spreadsheet(self, title: str, folder_id: Optional[str] = None) -> Tuple[str, str]:
        body = {"name": title, "mimeType": "application/vnd.google-apps.spreadsheet"}
        if folder_id:
            body["parents"] = [folder_id]
        new_ss = self._execute_with_backoff(
            self.clients.drive.files().create(body=body, supportsAllDrives=True, fields="id"),
            label="files.create[ss]",
        )
        spreadsheet_id = new_ss.get("id")
        url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
        self._dbg(f"[SS] created id={spreadsheet_id} title={title}")
        return spreadsheet_id, url

    def write_rows_to_new_spreadsheet(
        self,
        title: str,
        rows: List[List[str]],
        *,
        folder_id: Optional[str] = None,
        sheet_title: str = "シート1",
    ) -> Tuple[str, str]:
        ss_id, url = self.create_basic_spreadsheet(title, folder_id)
        # 初期シートのタイトル変更 (オプション)
        try:
            if sheet_title and sheet_title != "シート1":
                meta = self._execute_with_backoff(
                    self.clients.sheets.spreadsheets().get(spreadsheetId=ss_id, fields="sheets.properties"),
                    label="spreadsheets.get[rename-init]",
                )
                first_id = meta["sheets"][0]["properties"]["sheetId"]
                self._execute_with_backoff(
                    self.clients.sheets.spreadsheets().batchUpdate(
                        spreadsheetId=ss_id,
                        body={"requests": [{"updateSheetProperties": {"properties": {"sheetId": first_id, "title": sheet_title}, "fields": "title"}}]},
                    ),
                    label="spreadsheets.batchUpdate[rename-init]",
                )
        except Exception:
            pass
        if rows:
            self.update_values(ss_id, f"'{sheet_title}'!A1", rows, label="values.update[snapshot]")
        return ss_id, url

    def _get_sheet_title_by_gid(self, spreadsheet_id: str, sheet_gid: int) -> Optional[str]:
        try:
            meta = self._execute_with_backoff(
                self.clients.sheets.spreadsheets().get(
                    spreadsheetId=spreadsheet_id, fields="sheets.properties"
                ),
                label="spreadsheets.get[sheetTitle]",
            )
            for sh in meta.get("sheets", []):
                props = sh.get("properties", {})
                if props.get("sheetId") == sheet_gid:
                    return props.get("title")
        except Exception as exc:
            self._dbg(f"[IRREGULAR][WARN] title by gid failed {exc}")
        return None

    def load_irregular_destinations(self) -> Tuple[Dict[str, str], Dict[Tuple[str, str], str]]:
        """
        Load irregular destination mappings.
        Returns:
            (maker_default_map, maker_code_map)
            maker_default_map: {メーカー: 依頼先} for rows where 商品CD is empty
            maker_code_map: {(メーカー, 商品CD_norm): 依頼先} for rows where 商品CD is present
        """
        s_id = getattr(self.settings, "irregular_dest_spreadsheet_id", None)
        gid = getattr(self.settings, "irregular_dest_gid", None)
        if not s_id or gid is None:
            self._dbg("[IRREGULAR] not configured")
            return {}, {}
        title = self._get_sheet_title_by_gid(s_id, int(gid))
        if not title:
            self._dbg("[IRREGULAR][WARN] sheet title not found")
            return {}, {}
        rng = f"'{title}'!C3:E"  # C=メーカー, D=商品CD, E=依頼先
        try:
            res = self._execute_with_backoff(
                self.clients.sheets.spreadsheets().values().get(
                    spreadsheetId=s_id,
                    range=rng,
                    valueRenderOption="FORMATTED_VALUE",
                ),
                label="values.get[irregular]",
            )
            values = res.get("values", [])
        except Exception as exc:
            self._dbg(f"[IRREGULAR][WARN] load failed {exc}")
            values = []
        maker_default: Dict[str, str] = {}
        maker_code: Dict[Tuple[str, str], str] = {}
        for row in values:
            maker = (row[0] if len(row) > 0 else "").strip()
            if not maker:
                break  # 終端: C列が空になったら以降はデータ無し
            code = (row[1] if len(row) > 1 else "").strip()
            dest = (row[2] if len(row) > 2 else "").strip()
            if not dest:
                continue
            if not code:
                if maker not in maker_default:
                    maker_default[maker] = dest
            else:
                norm = code.lstrip("0") or "0"
                maker_code[(maker, norm)] = dest
                maker_code[(maker, code)] = dest
        self._dbg(f"[IRREGULAR] default={len(maker_default)} specific={len(maker_code)}")
        return maker_default, maker_code

    def create_extraction_sheet(
        self,
        *,
        center_id: str,
        flags_list: FlagsList,
        folder_id: Optional[str] = None,
        title_prefix: str = "抽出結果",
    ) -> Tuple[str, str]:
        title = f"{title_prefix}_{int(time.time())}_{center_id}"
        ss_id, url = self.create_basic_spreadsheet(title, folder_id)
        header = ["依頼先", "メーカー", "商品CD", "成分表", "見本"]
        rows = [header]
        maker_default, maker_code = self.load_irregular_destinations()
        for maker, code, s_flag, m_flag in flags_list:
            maker_key = (maker or "").strip() or "メーカー名なし"
            code_key = (code or "").strip()
            code_norm = code_key.lstrip("0") or "0"
            dest = maker_default.get(maker_key)
            if dest is None:
                dest = maker_code.get((maker_key, code_norm)) or maker_code.get((maker_key, code_key))
            if not dest:
                dest = maker_key
            rows.append([dest, maker_key, code_key, s_flag, m_flag])
        self.update_values(ss_id, "A1", rows, label="values.update[extraction]")
        self._dbg(f"[EXTRACT] sheet id={ss_id} rows={len(rows)-1}")
        return ss_id, url

    def load_extraction_sheet(self, spreadsheet_id: str, sheet_name: Optional[str] = None) -> FlagsList:
        rng = f"{sheet_name}!A1:Z" if sheet_name else "A1:Z"
        res = self._execute_with_backoff(
            self.clients.sheets.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=rng,
                valueRenderOption="FORMATTED_VALUE",
            ),
            label="values.get[extraction]",
        )
        values = res.get("values", [])
        if not values:
            return []
        header = values[0]
        # 依頼先列 (A列) はあってもなくても良い。あれば index を取っておく。
        try:
            dest_idx = header.index("依頼先")
        except ValueError:
            dest_idx = -1
        try:
            maker_idx = header.index("メーカー")
            cd_idx = header.index("商品CD")
        except ValueError:
            self._dbg("[EXTRACT][WARN] header missing required columns")
            return []
        # 成分表/見本 の列名互換 (見本/サンプル)
        try:
            seibun_idx = header.index("成分表")
        except ValueError:
            seibun_idx = -1
        sample_candidates = [c for c in ("見本", "サンプル") if c in header]
        sample_idx = header.index(sample_candidates[0]) if sample_candidates else -1
        out: FlagsList = []
        for row in values[1:]:
            if len(row) <= max(maker_idx, cd_idx):
                continue
            dest = (row[dest_idx] or "").strip() if dest_idx >= 0 and dest_idx < len(row) else ""
            maker = (row[maker_idx] or "").strip()
            code = (row[cd_idx] or "").strip()
            if not maker and not code:
                continue
            s_flag = (row[seibun_idx].strip() if seibun_idx >= 0 and seibun_idx < len(row) else "") if seibun_idx >= 0 else ""
            m_flag_raw = (row[sample_idx].strip() if sample_idx >= 0 and sample_idx < len(row) else "") if sample_idx >= 0 else ""
            # 正規化: 成分表は "○" のみ、見本は "3" or "○" をそのまま保持
            if s_flag not in {"", "○", "-"}:
                s_flag = "○" if s_flag else ""
            if m_flag_raw not in {"", "3", "○", "-"}:
                # 人が編集した場合の緩和 (例: '◯')
                m_flag = "3" if m_flag_raw == "3" else ("○" if m_flag_raw else "")
            else:
                m_flag = m_flag_raw
            out.append([dest, (maker or "メーカー名なし"), code, s_flag, m_flag])
        self._dbg(f"[EXTRACT] loaded rows={len(out)}")
        return out
    def load_product_catalog(self, spreadsheet_id: str, a1_range: str) -> Dict[str, List[str]]:
        response = self._execute_with_backoff(
            self.clients.sheets.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=a1_range,
                valueRenderOption="FORMATTED_VALUE",
            ),
            label="values.get[catalog]",
        )
        values = response.get("values", [])
        catalog: Dict[str, List[str]] = {}
        for row in values:
            if not row:
                continue
            code_raw = (row[0] or "").strip()
            if not code_raw:
                continue
            code_norm = code_raw.lstrip("0") or "0"
            catalog[code_norm] = row
            catalog[code_raw] = row
        self._dbg(f"[CATALOG] size={len(catalog)}")
        return catalog

    def write_ocr_snapshot(self, sheet_name: str, rows: List[List[str]]) -> str:
        sheet_conf = self.settings.sheets.get(sheet_name, None)
        if not sheet_conf:
            return ""
        self.clear_sheet(sheet_conf.spreadsheet_id, f"{sheet_conf.range_prefix}A1:ZZ")
        if rows:
            self.update_values(
                sheet_conf.spreadsheet_id,
                f"{sheet_conf.range_prefix}A1",
                rows,
                label="values.update(main)",
            )
        gid = sheet_conf.gid
        url = f"https://docs.google.com/spreadsheets/d/{sheet_conf.spreadsheet_id}/edit#gid={gid}"
        self._dbg(f"[OCR] snapshot url={url}")
        return url

    # ------------------------------------------------------------------
    def _col_letter(self, index_zero: int) -> str:
        n = index_zero + 1
        letters = ""
        while n:
            n, remainder = divmod(n - 1, 26)
            letters = chr(65 + remainder) + letters
        return letters

    def poll_until_ready(
        self,
        spreadsheet_id: str,
        start_row: int,
        n_rows: int,
        end_col_char: str,
        *,
        ready_col_idx: int,
        poll_min_ready: float,
        poll_max_wait: float,
        start_col_char: str = "B",
    ) -> List[List[str]]:
        rng = [f"'成分表'!{start_col_char}{start_row}:{end_col_char}{start_row + n_rows - 1}"]
        expected_width = ord(end_col_char.upper()) - ord(start_col_char.upper()) + 1
        t0 = time.time()
        delay = 0.20
        attempt = 0
        last_ready = -1
        stable = 0
        MIN_STABLE = 3

        while True:
            attempt += 1
            res = self.batch_get_values(spreadsheet_id, rng, label="values.batchGet[poll]")
            values = (res.get("valueRanges") or [{}])[0].get("values", [])
            if len(values) < n_rows:
                values.extend([[] for _ in range(n_rows - len(values))])
            values = [(row + [""] * (expected_width - len(row)))[:expected_width] for row in values]
            ready_rows = sum(1 for row in values if str(row[ready_col_idx]).strip())
            ready_ratio = ready_rows / max(1, n_rows)
            self._dbg(f"[POLL] ready={ready_rows}/{n_rows} ({ready_ratio:.2f}) attempt={attempt} delay={delay:.2f}s")
            if ready_ratio >= poll_min_ready:
                return values
            if ready_rows == last_ready:
                stable += 1
            else:
                stable = 0
                last_ready = ready_rows
            if stable >= MIN_STABLE and ready_ratio >= 0.5:
                return values
            if time.time() - t0 > poll_max_wait:
                return values
            time.sleep(delay + random.uniform(0, delay * 0.3))
            delay = min(delay * 2, 5)

    def compute_all_in_one_copy(
        self,
        work_id: str,
        selections: Sequence[Tuple[str, str, str, str]],
        start_row: int,
        end_col_char: str,
        *,
        chunk_size: int,
        ready_col_idx: int,
        poll_min_ready: float,
        poll_max_wait: float,
        start_col_char: str = "B",
    ) -> List[List[str]]:
        values_all: List[List[str]] = []
        idx = 0
        total = len(selections)
        while idx < total:
            batch = selections[idx : idx + chunk_size]
            cds = [[cd] for (_, cd, _, _) in batch]
            self.batch_update_values(
                work_id,
                [
                    {
                        "range": f"'成分表'!A{start_row}",
                        "values": cds,
                    }
                ],
            )
            vals = self.poll_until_ready(
                work_id,
                start_row,
                len(batch),
                end_col_char,
                ready_col_idx=ready_col_idx,
                poll_min_ready=poll_min_ready,
                poll_max_wait=poll_max_wait,
                start_col_char=start_col_char,
            )
            values_all.extend(vals)
            self.clear_sheet(work_id, f"'成分表'!A{start_row}:A{start_row + len(batch) + 10}")
            idx += len(batch)
        return values_all

    # ------------------------------------------------------------------
    def create_work_copy(self, original_id: str, title: str) -> str:
        body = {"name": title}
        copy = self._execute_with_backoff(
            self.clients.drive.files().copy(
                fileId=original_id, body=body, supportsAllDrives=True, fields="id"
            ),
            label="files.copy",
        )
        return copy.get("id")  # type: ignore[return-value]

    def delete_file(self, file_id: str) -> None:
        try:
            self._execute_with_backoff(
                self.clients.drive.files().delete(fileId=file_id, supportsAllDrives=True),
                label="files.delete",
            )
        except Exception as exc:
            self._dbg(f"[WARN] files.delete {file_id} {exc}")

    # ------------------------------------------------------------------
    def build_maker_rows(
        self,
        catalog: Dict[str, List[str]],
        maker_cds: MakerCodes,
    ) -> MakerData:
        maker_data: MakerData = {}
        for maker, cds in maker_cds.items():
            rows: List[List[str]] = []
            miss = 0
            for code in cds:
                row = catalog.get(code) or catalog.get(code.lstrip("0") or "0")
                if row:
                    maker_val = row[1] if len(row) > 1 else maker
                    product_name = row[2] if len(row) > 2 else ""
                    spec = row[3] if len(row) > 3 else ""
                    maker_product_cd = row[4] if len(row) > 4 else ""
                    jan = row[5] if len(row) > 5 else ""
                    note = row[7] if len(row) > 7 else ""
                    # 行構成: [メーカー, 商品名, 規格, 備考, メーカー商品CD, JAN]
                    rows.append([maker_val, product_name, spec, note, maker_product_cd, jan])
                else:
                    miss += 1
                    rows.append([maker, "", "", f"NOT_FOUND:{code}", "", ""])
            if miss:
                self._dbg(f"[CATALOG][MISS] maker={maker} missing={miss}/{len(cds)}")
            maker_data[maker] = rows
        return maker_data

    # ------------------------------------------------------------------
    def create_output_spreadsheet(
        self,
        title: str,
        *,
        drive_folder_id: str,
    ) -> Tuple[str, str]:
        body = {
            "name": title,
            "mimeType": "application/vnd.google-apps.spreadsheet",
        }
        if drive_folder_id:
            body["parents"] = [drive_folder_id]
        new_ss = self._execute_with_backoff(
            self.clients.drive.files().create(
                body=body,
                supportsAllDrives=True,
            ),
            label="files.create",
        )
        spreadsheet_id = new_ss.get("id")
        url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
        return spreadsheet_id, url

    def delete_default_sheet(self, spreadsheet_id: str) -> None:
        try:
            meta = self._execute_with_backoff(
                self.clients.sheets.spreadsheets().get(
                    spreadsheetId=spreadsheet_id, fields="sheets.properties"
                ),
                label="spreadsheets.get",
            )
            first_id = meta["sheets"][0]["properties"]["sheetId"]
            self._execute_with_backoff(
                self.clients.sheets.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={"requests": [{"deleteSheet": {"sheetId": first_id}}]},
                ),
                label="spreadsheets.batchUpdate[delFirst]",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    def generate_documents(
        self,
        maker_data: MakerData,
        maker_cds: MakerCodes,
        flags_list: FlagsList,
        *,
        doc_title: Optional[str] = None,
        center_name: str,
        center_month: str,
        center_conf: Dict,
        center_id: str,
        output_folder_id: Optional[str] = None,
        existing_spreadsheet_id: Optional[str] = None,
        existing_url: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        center_conf = center_conf or {}
        self.debug_events = []
        # 既存スプレッドシートが指定されている場合はそれを使う（シート追加のみ）
        if existing_spreadsheet_id:
            spreadsheet_id = existing_spreadsheet_id
            url = existing_url or f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
            self._dbg(f"[STEP2] reuse existing spreadsheet id={spreadsheet_id} center_id={center_id}")
        else:
            title = f"依頼書出力_{int(time.time())}"
            self._dbg(f"[STEP2] create {title} center_id={center_id}")
        if os.environ.get("DRIVE_DIAG", "1") == "1":
            try:
                sa_email = getattr(self.clients.credentials, "service_account_email", "UNKNOWN")
                self._dbg(f"[STEP2][DIAG] sa_email={sa_email}")
            except Exception as exc:
                self._dbg(f"[STEP2][DIAG][WARN] sa_email fetch failed {exc}")
            try:
                about = self.clients.drive.about().get(fields="storageQuota").execute()
                quota = about.get("storageQuota", {})
                usage = quota.get("usage")
                limit = quota.get("limit")
                self._dbg(f"[STEP2][DIAG] usage={usage} limit={limit}")
            except Exception as exc:
                self._dbg(f"[STEP2][DIAG][WARN] about.get failed {exc}")
            try:
                q = "name contains '依頼書出力_' and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
                res = self.clients.drive.files().list(
                    q=q,
                    pageSize=10,
                    fields="files(id,name)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                ).execute()
                names = [f.get("name") for f in res.get("files", [])][:3]
                self._dbg(f"[STEP2][DIAG] recent_outputs={len(res.get('files', []))} sample={names}")
            except Exception as exc:
                self._dbg(f"[STEP2][DIAG][WARN] list failed {exc}")

        target_folder = output_folder_id or self.settings.drive_folder_id
        if not existing_spreadsheet_id:
            spreadsheet_id, url = self.create_output_spreadsheet(
                title=title,
                drive_folder_id=target_folder,
            )
            self._dbg(f"[STEP2] new_id={spreadsheet_id}")

        template_id_present = "templateSpreadsheetId" in center_conf
        template_sheet_id_present = "templateSheetId" in center_conf
        export_start_row_present = "exportStartRow" in center_conf
        template_id = center_conf.get("templateSpreadsheetId") if template_id_present else self.settings.template_spreadsheet_id
        template_sheet_id = center_conf.get("templateSheetId") if template_sheet_id_present else self.settings.template_sheet_id
        start_row = center_conf.get("exportStartRow", self.settings.start_row)
        self._dbg(
            f"[CFG] templateSpreadsheetId source={'center' if template_id_present else 'default'} value={template_id}"
        )
        self._dbg(
            f"[CFG] templateSheetId source={'center' if template_sheet_id_present else 'default'} value={template_sheet_id}"
        )
        self._dbg(
            f"[CFG] exportStartRow source={'center' if export_start_row_present else 'default'} value={start_row}"
        )
        poll_conf = center_conf.get("poll") if center_conf else None
        if poll_conf:
            self._dbg(f"[CFG] poll (center) {poll_conf}")
        else:
            self._dbg("[CFG] poll default env values")

        if not template_id or not template_sheet_id:
            raise RuntimeError("テンプレートID/シートIDが未設定です (center 設定または環境変数を確認してください)")

        export_ranges = self.settings.export_ranges()
        center_export_ranges = center_conf.get("ranges", {}).get("export", {}) if center_conf else {}
        maker_header_rng = center_export_ranges.get("makerHeader", export_ranges["makerHeader"])
        center_name_rng = center_export_ranges.get("centerName", export_ranges["centerName"])
        month_rng = center_export_ranges.get("month", export_ranges["month"])
        if center_export_ranges:
            self._dbg(f"[CFG] export ranges center(specified)={center_export_ranges}")
        else:
            self._dbg(f"[CFG] export ranges default={export_ranges}")

        flags_map = {(maker, cd): (s_flag, m_flag) for maker, cd, s_flag, m_flag in flags_list}

        # 依頼先ごとにシートは1枚とするため、maker_data 全体を1シートに書き出す
        # 行をフラットにし、メーカー列を含めたまま1シートに展開する
        all_rows: List[List[str]] = []
        all_cds: List[str] = []
        for maker, rows in maker_data.items():
            cds_for_maker = maker_cds.get(maker, [])
            n = min(len(cds_for_maker), len(rows))
            if n <= 0:
                continue
            cds_slice = cds_for_maker[:n]
            rows_slice = rows[:n]
            all_cds.extend(cds_slice)
            all_rows.extend(rows_slice)

        if not all_rows:
            self._dbg("[MK-SHEET] no rows to write for this document")
            return url, self.debug_events

        copied = self._execute_with_backoff(
            self.clients.sheets.spreadsheets().sheets().copyTo(
                spreadsheetId=template_id,
                sheetId=template_sheet_id,
                body={"destinationSpreadsheetId": spreadsheet_id},
            ),
            label="sheets.copyTo",
        )
        new_sheet_id = copied.get("sheetId")
        # シート名は依頼書タイトル（依頼先名など）を優先して使う
        safe_title = self._sanitize_title(doc_title or center_name or "依頼書", set())
        self._execute_with_backoff(
            self.clients.sheets.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={
                    "requests": [
                        {
                            "updateSheetProperties": {
                                "properties": {"sheetId": new_sheet_id, "title": safe_title},
                                "fields": "title",
                            }
                        }
                    ]
                },
            ),
            label="spreadsheets.batchUpdate[rename]",
        )

        # ヘッダ情報（センター名・月）はシート共通
        if center_name:
            self.update_values(
                spreadsheet_id,
                f"'{safe_title}'!{center_name_rng}",
                [[center_name] * 4],
                label="values.update[センター名]",
            )
        if center_month:
            self.update_values(
                spreadsheet_id,
                f"'{safe_title}'!{month_rng}",
                [[center_month]],
                label="values.update[月]",
            )

        # ヘッダ行から列位置を検出
        header_row = start_row - 1
        rng_header = f"'{safe_title}'!A{header_row}:Z{header_row}"
        res_header = self._execute_with_backoff(
            self.clients.sheets.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=rng_header,
                valueRenderOption="FORMATTED_VALUE",
            ),
            label="values.get[export-header]",
        )
        header_vals = (res_header.get("values") or [[]])[0]

        def col_letter(name: str, default_idx: int) -> str:
            if name in header_vals:
                idx = header_vals.index(name)
                return self._col_letter(idx)
            return self._col_letter(default_idx)

        # 列位置の検出
        spec_col = col_letter("規格", 7)                # 既定: H 列
        seibun_col_letter = col_letter("成分表", 8)     # 既定: I 列
        # サンプルは見出し互換（サンプル or 見本）。既定: J 列
        sample_col_letter: str
        if "サンプル" in header_vals:
            sample_col_letter = self._col_letter(header_vals.index("サンプル"))
        elif "見本" in header_vals:
            sample_col_letter = self._col_letter(header_vals.index("見本"))
        else:
            sample_col_letter = self._col_letter(9)
        maker_cd_col = col_letter("メーカー商品CD", 5)  # 既定: F 列
        jan_col = col_letter("JAN", 6)                 # 既定: G 列
        biko_cols = [idx for idx, val in enumerate(header_vals) if str(val).startswith("備考")]
        # 備考1列目、無ければ J の次列(K) を使用
        note_col = self._col_letter(biko_cols[0]) if biko_cols else self._col_letter(ord(sample_col_letter) - 65 + 1)
        # 備考2列目、無ければ 前項の次列(L) を使用
        note2_col = self._col_letter(biko_cols[1]) if len(biko_cols) > 1 else self._col_letter(ord(note_col) - 65 + 1)

        # 値生成（テンプレ式に頼らず値で書く）
        makers_col = [[row[0] if len(row) > 0 else ""] for row in all_rows]
        product_col = [[row[1] if len(row) > 1 else ""] for row in all_rows]
        spec_values = [[row[2] if len(row) > 2 else ""] for row in all_rows]
        notes = [[row[3] if len(row) > 3 else ""] for row in all_rows]
        maker_cd_values = [[row[4] if len(row) > 4 else ""] for row in all_rows]
        jan_values = [[row[5] if len(row) > 5 else ""] for row in all_rows]

        seibun_values: List[List[str]] = []
        mihon_values: List[List[str]] = []
        for maker, code, _, _ in flags_list:
            s_flag, m_flag = flags_map.get((maker, code), ("", ""))
            seibun_values.append([s_flag])
            mihon_values.append([m_flag])

        n = min(len(all_cds), len(all_rows))
        all_cds = all_cds[:n]
        makers_col = makers_col[:n]
        product_col = product_col[:n]
        spec_values = spec_values[:n]
        notes = notes[:n]
        maker_cd_values = maker_cd_values[:n]
        jan_values = jan_values[:n]
        seibun_values = seibun_values[:n]
        mihon_values = mihon_values[:n]

        data_updates = [
            {"range": f"'{safe_title}'!A{start_row}", "values": [[code] for code in all_cds]},
            {"range": f"'{safe_title}'!C{start_row}", "values": makers_col},              # C: メーカー
            {"range": f"'{safe_title}'!D{start_row}", "values": product_col},             # D: 商品名
            {"range": f"'{safe_title}'!{maker_cd_col}{start_row}", "values": maker_cd_values},
            {"range": f"'{safe_title}'!{jan_col}{start_row}", "values": jan_values},
            {"range": f"'{safe_title}'!{spec_col}{start_row}", "values": spec_values},
            {"range": f"'{safe_title}'!{seibun_col_letter}{start_row}", "values": seibun_values},
            {"range": f"'{safe_title}'!{sample_col_letter}{start_row}", "values": mihon_values},
            {"range": f"'{safe_title}'!{note_col}{start_row}", "values": notes},
            {"range": f"'{safe_title}'!{note2_col}{start_row}", "values": [[""]] * n},
        ]
        self.batch_update_values(spreadsheet_id, data_updates)
        self._dbg("[MK-SHEET] template formulas preserved: write A/I/J only")
