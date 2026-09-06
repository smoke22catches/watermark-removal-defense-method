# Run artifacts

Each training or evaluation invocation creates an independent timestamped directory:

```
runs/<YYYYmmdd-HHMMSS>_<run-name>/
├── config.yaml      # full resolved config snapshot
├── metrics.csv      # per-epoch (train) or per-attack (eval) metrics
├── checkpoints/     # best.pt, last.pt (training only)
├── plots/           # loss curves, bit-accuracy charts, qualitative grids
└── log.txt          # console log mirror
```

Runs are never overwritten: a new timestamp is always allocated.
This directory (except this README) is gitignored.
