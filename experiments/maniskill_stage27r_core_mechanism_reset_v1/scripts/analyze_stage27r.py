#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import PROTOCOL_ID, atomic_json


def ci(values, seed=2718001, n=10000):
    x=np.asarray(values,float); rng=np.random.default_rng(seed); means=np.empty(n)
    for i in range(n): means[i]=np.mean(x[rng.integers(0,len(x),len(x))])
    return {"mean":float(np.mean(x)),"ci95":[float(np.quantile(means,.025)),float(np.quantile(means,.975))]}


def signflip(values,n=10000,seed=2718002):
    x=np.asarray(values,float); observed=abs(float(np.mean(x))); rng=np.random.default_rng(seed); exceed=0
    for _ in range(n): exceed += abs(float(np.mean(x*rng.choice([-1,1],len(x)))))>=observed
    return (exceed+1)/(n+1)


def aggregate(raw):
    grouped=defaultdict(list)
    for row in raw: grouped[(row["task"],row["model_seed"],row["bank_id"],row["source_episode"],row["phase"],row["condition"])].append(row)
    result=[]
    for key,rows in grouped.items():
      base={name:value for name,value in zip(["task","model_seed","bank_id","source_episode","phase","condition"],key)}
      base.update(success_hold5=float(np.mean([r["success_hold5"] for r in rows])),normalized_progress=float(np.mean([r["normalized_progress"] for r in rows])),utility={w:float(np.mean([r["utilities"][w] for r in rows])) for w in rows[0]["utilities"]},cost=float(np.mean([r["accounting"]["estimated_flops"] for r in rows])),causal=all(r["causal_fidelity_pass"] for r in rows))
      result.append(base)
    return result


def state_table(rows,weight):
    grouped=defaultdict(dict)
    for r in rows: grouped[(r["task"],r["model_seed"],r["bank_id"],r["source_episode"],r["phase"])][r["condition"]]=r
    out=[]
    for key,c in grouped.items():
      fc_rows=[value for name,value in c.items() if name.startswith("FC_tile")]; ff_rows=[value for name,value in c.items() if name.startswith("FF_tile")]
      if not fc_rows or len(fc_rows)!=len(ff_rows) or "CC" not in c or "CF" not in c: raise RuntimeError(f"incomplete factorial {key}: {len(c)}")
      fc=max(fc_rows,key=lambda r:(r["utility"][weight],-int(r["condition"].split("tile")[-1])))
      ff=max(ff_rows,key=lambda r:(r["utility"][weight],-int(r["condition"].split("tile")[-1])))
      cc,cf=c["CC"],c["CF"]
      out.append({"key":key,"task":key[0],"seed":key[1],"source_episode":key[3],"phase":key[4],"CC":cc,"CF":cf,"FC":fc,"FF":ff,"dv":fc["utility"][weight]-cc["utility"][weight],"da":cf["utility"][weight]-cc["utility"][weight],"dj":ff["utility"][weight]-max(fc["utility"][weight],cf["utility"][weight])})
    return out


def arm_allocate(states,budget_fraction,weight):
    coarse=sum(s["CC"]["cost"] for s in states); full=sum(s["FF"]["cost"] for s in states); budget=budget_fraction*full
    def allocate(options):
      chosen=["CC"]*len(states); cost=coarse
      upgrades=[]
      for i,s in enumerate(states):
       for mode in options:
        dc=s[mode]["cost"]-s["CC"]["cost"]; du=s[mode]["utility"][weight]-s["CC"]["utility"][weight]
        upgrades.append((du/max(dc,1),du,-dc,-i,i,mode,dc))
      for _,du,_,_,i,mode,dc in sorted(upgrades,reverse=True):
       if chosen[i]!="CC" or du<=0 or cost+dc>budget: continue
       chosen[i]=mode; cost+=dc
      return chosen,cost
    aggregate_axis="FC" if np.mean([s["dv"] for s in states])>=np.mean([s["da"] for s in states]) else "CF"
    arms={"all_coarse":(["CC"]*len(states),coarse),"all_fine":(["FF"]*len(states),full),"visual_only_oracle":allocate(["FC"]),"action_only_oracle":allocate(["CF"]),"strongest_equal_cost_fixed_axis":allocate([aggregate_axis]),"state_axis_oracle":allocate(["FC","CF"]),"joint_oracle":allocate(["FC","CF","FF"])}
    order=sorted(range(len(states)),key=lambda i:hashlib.sha256(str(states[i]["key"]).encode()).hexdigest()); random=["CC"]*len(states); cost=coarse
    for i in order:
      dc=states[i]["FF"]["cost"]-states[i]["CC"]["cost"]
      if cost+dc<=budget: random[i]="FF"; cost+=dc
    arms["random_state"]=(random,cost)
    phase_order=sorted(range(len(states)),key=lambda i:(states[i]["phase"]!="contact_placement_near_completion",states[i]["phase"]!="object_in_hand_pre_placement",i)); heuristic=["CC"]*len(states); cost=coarse
    for i in phase_order:
      dc=states[i]["FF"]["cost"]-states[i]["CC"]["cost"]
      if cost+dc<=budget: heuristic[i]="FF"; cost+=dc
    arms["phase_heuristic"]=(heuristic,cost)
    result={}
    for name,(modes,cost) in arms.items():
      result[name]={"success_hold5":float(np.mean([s[m]["success_hold5"] for s,m in zip(states,modes)])),"utility":float(np.mean([s[m]["utility"][weight] for s,m in zip(states,modes)])),"cost":cost,"budget":budget,"budget_compliant":bool(cost<=budget+1e-6),"refined_states":sum(m!="CC" for m in modes)}
    return result


def main():
 p=argparse.ArgumentParser(); p.add_argument("--inputs",type=Path,nargs="+",required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); raw=[]
 for path in a.inputs: raw.extend(json.loads(path.read_text())["rows"])
 rows=aggregate(raw); stats={}; budgets={}
 for weight in ("balanced","success_dominant","progress_dominant"):
  states=state_table(rows,weight); stats[weight]={}
  for task in sorted(set(s["task"] for s in states)):
   task_states=[s for s in states if s["task"]==task]
   stats[weight][task]={effect:{**ci([s[field] for s in task_states]),"signflip_p":signflip([s[field] for s in task_states])} for effect,field in [("visual","dv"),("action","da"),("joint","dj")]}
   stats[weight][task]["positive_joint_state_fraction"]=float(np.mean([s["dj"]>0 for s in task_states]))
  budgets[weight]={str(fraction):arm_allocate(states,fraction,weight) for fraction in (.25,.50,.75)}
 atomic_json(a.output,{"protocol_id":PROTOCOL_ID,"aggregated_state_treatments":rows,"statistics":stats,"budgets":budgets,"bootstrap_replicates":10000,"primary_unit":"source_episode"})

if __name__=="__main__": main()
