import subprocess
import csv
import io
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import math
import random
import time

@dataclass(frozen=True)
class Config:
    BM: int
    BN: int
    BK: int
    U: int
    T: int

def run_matmul(cfg: Config, *, M=1024, N=1024, K=1024, repeat=5, exe="./matmul") -> float:
    """
    ブラックボックス評価：C++バイナリを実行して median_ms を返す
    失敗したら inf を返す
    """
    cmd = [
        exe,
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--BM", str(cfg.BM), "--BN", str(cfg.BN), "--BK", str(cfg.BK),
        "--U", str(cfg.U), "--T", str(cfg.T),
        "--repeat", str(repeat),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        # コンパイル/実行失敗＝制約違反扱い（または測定不能）
        # print(e.output)
        return float("inf")

    # 出力は2行：ヘッダ + 1データ行（CSV）
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return float("inf")

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    row = next(reader, None)
    if row is None:
        return float("inf")

    return float(row["median_ms"])

def make_candidates() -> List[Config]:
    BMs = [8, 16, 32, 48, 64, 96, 128]
    BNs = [8, 16, 32, 48, 64, 96, 128]
    BKs = [4, 8, 16, 24, 32, 48]
    Us = [1, 2, 4, 8]
    Ts = [1, 2, 3, 4]
 # Macのコア数に合わせて後で調整してOK

    cands = []
    for BM in BMs:
        for BN in BNs:
            for BK in BKs:
                for U in Us:
                    for T in Ts:
                        cands.append(Config(BM, BN, BK, U, T))
    return cands
