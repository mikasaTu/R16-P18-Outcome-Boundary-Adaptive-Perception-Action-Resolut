# Raw evidence manifest

The raw evidence is 5.6 GiB of counterfactual data, 280 MiB of closed-loop
traces, and 7.3 MiB of predictor artifacts. It remains on CPFS at:

`/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage26-counterfactual-completion-v1/r16p18-stage26-counterfactual-20260816-v13`

The v13 paths are immutable read-only links to completed v10 collection and
fidelity evidence and completed v12 predictor/closed-loop evidence. The compact
Git snapshot contains all decision outputs, fidelity summaries, predictor
freeze, and all 21 arm summaries.

Counts:

- completed collection shards: 120
- collection-complete banks: 6
- rollout capsules: 7,565
- counterfactual branch rows: 22,695
- shared-prefix comparisons: 1,920
- closed-loop episode rows: 4,200
- closed-loop complete arm/seed cells: 21

Collection marker SHA256 values:

- seed 16018 calibration: `99dc266ae2612efba21fc9c7f456d51c781a3ac95fa3d4b52917e1b3eb561ecc`
- seed 16018 train: `e151c4afa0ff72fb62a5ecd1e2e888e26f511f5233fd17124503cb975b472bae`
- seed 16019 calibration: `c42bad8270d3f5a1eadd8be39aa9e345828f4c38df3812932312a306f5ac4296`
- seed 16019 train: `4989d62115e6b0858fa8a0132b471a55250d11ef1783725a8f0784b40fc77c5e`
- seed 16020 calibration: `5c350b507b3656eef5157d4116a68e76a6016f4186cc0f5dcd40dbbd6458f0d0`
- seed 16020 train: `079e21198675c45e2ccef3227b06e7cff4808abdd481f0bd338e73d50418f77c`

Key v13 hashes:

- `FORMAL_RUN_MANIFEST.json`: `899036a3794a4038cb3b1f772a5e02b13beda003e2e5129f98aa5e003bcd3ddc`
- `FORMAL_COMPLETE.json`: `7b73af8682523d803a897c57990ab5bf30b687d61cd5d4049a553cd0392fad88`
- `stage26_summary.json`: `9617b8e812e3e1b7e46d6eee50a7a9b68bc9e535cd36fb91215ec18d97755a44`
- `independent_stage26_audit.json`: `b3aa40398393db28bf8c50a1235d325f0df7722eae0ff4e8ec82e393e8916ba0`
- `extended_statistics.json`: `ec234a5a794c1de8e0a43778d3256f2a46721f676b80ebba56f01b3cdf5d95d1`
- `mechanism_analysis.json`: `3201ca63a4bb5ac25dd9eb8c88cd6a79b68243db0833ed525938cc9769985a01`
