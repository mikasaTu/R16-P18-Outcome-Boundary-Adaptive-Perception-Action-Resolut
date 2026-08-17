#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from common import PROTOCOL_ID,atomic_json
def gain(paths):
 rows=[]
 for p in paths: rows.extend(json.loads(p.read_text())["rows"])
 grouped={}
 for r in rows: grouped.setdefault((r["model_seed"],r["bank_id"],r["condition"]),[]).append(r["utilities"]["balanced"])
 states={}
 for (seed,bank,condition),vals in grouped.items(): states.setdefault((seed,bank),{})[condition]=np.mean(vals)
 gains=[]
 for c in states.values(): gains.append(max(v for k,v in c.items() if k.startswith("FF_tile"))-c["CC"])
 return float(np.mean(gains))
def main():
 p=argparse.ArgumentParser();p.add_argument("--grid2",type=Path,nargs="+",required=True);p.add_argument("--grid4",type=Path,nargs="+",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();g2,g4=gain(a.grid2),gain(a.grid4);ratio=g2/g4 if g4>0 else (1.0 if g2>=g4 else float('-inf'));selected=2 if ratio>=.90 else 4;atomic_json(a.output,{"protocol_id":PROTOCOL_ID,"grid2_oracle_gain":g2,"grid4_oracle_gain":g4,"recovery_ratio":ratio,"threshold":.90,"selected_grid":selected})
if __name__=="__main__":main()
