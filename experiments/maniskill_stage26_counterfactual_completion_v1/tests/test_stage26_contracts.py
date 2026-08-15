from __future__ import annotations

import ast,json,random,sys
from collections import deque
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from common import write_json_new
from predictor import CompletionModel,FeatureShape
from stage26_runtime import Capsule,capsule_bytes,capture_rng,load_capsule,restore_rng,save_capsule_new

def capsule()->Capsule:
    return Capsule("id","first_success","StackCube-v1",16018,7,"/x","0"*64,12,"placement_contact_near_completion",{"a":np.array([1.])},12,{"state":np.zeros(2)},"a"*64,[[0.,1.]],[[2.,3.]],[[0.,0.,0.,1.]],[[0.,0.,0.,1.]]*5,np.zeros((2,3,4),np.float32),[0.,0.,0.,1.],1.,True,1,2,capture_rng(),"b"*64,[])

def test_capsule_serialization_and_act_prefix(tmp_path:Path)->None:
    value=capsule();path=tmp_path/"capsule.pt";digest=save_capsule_new(path,value);restored=load_capsule(path)
    assert len(digest)==64 and restored.capsule_id==value.capsule_id
    assert np.array_equal(restored.temporal_table_prefix,value.temporal_table_prefix)
    with pytest.raises(FileExistsError):save_capsule_new(path,value)

def test_rng_roundtrip()->None:
    random.seed(3);np.random.seed(3);torch.manual_seed(3);state=capture_rng();expected=(random.random(),np.random.rand(),torch.rand(1).item());restore_rng(state);actual=(random.random(),np.random.rand(),torch.rand(1).item());assert np.allclose(expected,actual)

def test_predictor_shapes()->None:
    shape=FeatureShape(8,3,4,20,4);x=torch.randn(5,shape.flat)
    for architecture in ("linear_probe","two_layer_mlp","one_layer_small_gru"):assert CompletionModel(architecture,shape)(x).shape==(5,3)

def test_seed_banks_pairwise_disjoint()->None:
    banks=json.loads((ROOT/"manifests/seed_banks.json").read_text())["banks"]
    values=[set(banks[k]) for k in ("train_source","calibration","confirmatory")]
    assert [len(v) for v in values]==[512,128,200]
    assert not values[0]&values[1] and not values[0]&values[2] and not values[1]&values[2]

def test_no_privileged_predictor_inputs()->None:
    text=(ROOT/"scripts/evaluate_closed_loop.py").read_text();tree=ast.parse(text)
    banned=("object_position","goal_distance","simulator_success","phase_truth","privileged_contact")
    feature_source=ast.get_source_segment(text,next(n for n in ast.walk(tree) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=="visual" for k in n.keys))) or ""
    assert all(word not in feature_source for word in banned)

def test_fail_on_overwrite(tmp_path:Path)->None:
    path=tmp_path/"x.json";write_json_new(path,{"x":1})
    with pytest.raises(FileExistsError):write_json_new(path,{"x":2})

def test_preemption_resume_marker_contract(tmp_path:Path)->None:
    marker=tmp_path/"SHARD_COMPLETE.json";write_json_new(marker,{"protocol_id":"R16-P18-MS5-STAGE26-COUNTERFACTUAL-COMPLETION-V1","episode_seeds":[1,2]})
    assert json.loads(marker.read_text())["episode_seeds"]==[1,2]

def test_partial_shard_recovery_preserves_evidence()->None:
    text=(ROOT/"scripts/collect_counterfactual_data.py").read_text()
    assert ".partial-preserved-" in text
    assert "shard.rename(preserved)" in text
