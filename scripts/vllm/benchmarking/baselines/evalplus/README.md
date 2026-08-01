# EvalPlus Baselines

Drop EvalPlus baselines here as `<config>.baseline.json`. To make one, redirect
`check_regression.py --mode evalplus --calibrate` output to a file. The nightly
job gates whichever baselines exist; the PR guard skips EvalPlus.
