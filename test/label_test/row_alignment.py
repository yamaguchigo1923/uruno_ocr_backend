from __future__ import annotations

from typing import List, Tuple


def score_number(gt: int, ocr: int) -> int:
    """番号同士のスコアを計算する。

    |gt-ocr|==0: +5, 1: +2, 2: +1, それ以外: -3
    """

    d = abs(gt - ocr)
    if d == 0:
        return 5
    if d == 1:
        return 2
    if d == 2:
        return 1
    return -3


def align_rows_by_number(
    gt_numbers: List[int],
    ocr_numbers: List[int],
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """番号のみを用いたシーケンスアラインメントで行マッチングを行う。

    入力:
        gt_numbers: 正解側番号配列 (長さ N)
        ocr_numbers: OCR側番号配列 (長さ M)

    出力:
        (pairs, skipped_gt_indices, skipped_ocr_indices)
    """

    n = len(gt_numbers)
    m = len(ocr_numbers)
    gap_penalty = -2

    # dp と trace を (n+1) x (m+1) で確保
    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    trace: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]

    # 初期化
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        trace[i][0] = 1  # 正解側スキップ
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
        trace[0][j] = 2  # OCR側スキップ

    # 遷移
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = dp[i - 1][j - 1] + score_number(gt_numbers[i - 1], ocr_numbers[j - 1])
            skip_gt = dp[i - 1][j] + gap_penalty
            skip_ocr = dp[i][j - 1] + gap_penalty

            best = match_score
            t = 0
            if skip_gt > best:
                best = skip_gt
                t = 1
            if skip_ocr > best:
                best = skip_ocr
                t = 2

            dp[i][j] = best
            trace[i][j] = t

    # 復元
    pairs_rev: List[Tuple[int, int]] = []
    skipped_gt_rev: List[int] = []
    skipped_ocr_rev: List[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        t = trace[i][j]
        if i > 0 and j > 0 and t == 0:
            # マッチ
            pairs_rev.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or t == 1):
            # 正解側スキップ
            skipped_gt_rev.append(i - 1)
            i -= 1
        elif j > 0 and (i == 0 or t == 2):
            # OCR側スキップ
            skipped_ocr_rev.append(j - 1)
            j -= 1
        else:
            # 保険: どれにも当てはまらない場合は両方デクリメント
            if i > 0:
                i -= 1
            if j > 0:
                j -= 1

    pairs_rev.reverse()
    skipped_gt_rev.reverse()
    skipped_ocr_rev.reverse()
    return pairs_rev, skipped_gt_rev, skipped_ocr_rev
