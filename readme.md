# Fair Transferability Subject to Bounded Distribution Shift

This repository accompanies the paper [Fair Transferability Subject to Bounded Distribution Shift](https://arxiv.org/abs/2206.00129) accepted by Neurips 2022. - [Yatong Chen](https://github.com/YatongChen/), [Reilly Raab](https://reillyraab.com/), [Jialu Wang](https://people.ucsc.edu/~jwang470/), [Yang Liu](http://www.yliuu.com/).

# Abstract: 
Given an algorithmic predictor that is "fair" on some source distribution, will it still be fair on an unknown target distribution that differs from the source within some bound? In this paper, we study the transferability of statistical group fairness for machine learning predictors (i.e., classifiers or regressors) subject to bounded distribution shift, a phenomenon frequently caused by user adaptation to a deployed model or a dynamic environment. Herein, we develop a bound characterizing such transferability, flagging potentially inappropriate deployments of machine learning for socially consequential tasks. We first develop a framework for bounding violations of statistical fairness subject to distribution shift, formulating a generic upper bound for transferred fairness violation as our primary result. We then develop bounds for specific worked examples, adopting two commonly used fairness definitions (i.e., demographic parity and equalized odds) for two classes of distribution shift (i.e., covariate shift and label shift). Finally, we compare our theoretical bounds to deterministic models of distribution shift as well as real-world data.

# Dependencies

Replicable Python environment
(using [conda](https://docs.conda.io/en/latest/miniconda.html))

```
├── environment.yml
```

$ conda env create -n fairtransfer --file environment.yml

# Figures 3a, 7, 8

Source data:

https://github.com/fairmlbook/fairmlbook.github.io/tree/master/code/creditscore/data

@book{barocas-hardt-narayanan,
  title = {Fairness and Machine Learning},
  author = {Solon Barocas and Moritz Hardt and Arvind Narayanan},
  publisher = {fairmlbook.org},
  note = {\url{http://www.fairmlbook.org}},
  year = {2019}
}

Instructions:

Run covariate/main.py. Intermediate values are saved as pickled arrays
for subsequent faster generation of plots.

```
├── covariate
│   ├── cdf_by_race.csv          -- source data
│   ├── default_by_race.csv      -- source data
│   ├── DP_bound.pdf             -- output (pdf)
│   ├── dp_max_bgs                 -- intermediate data (pickled array)
│   ├── dp_source_ac               -- intermediate data (pickled array)
│   ├── dp_strategic_manipulation  -- intermediate data (pickled array)
│   ├── dp_upper_bound             -- intermediate data (pickled array)
│   ├── EO_bound.pdf             -- output (pdf)
│   ├── main.py                  -- run this
│   ├── max_bgs                    -- intermediate data (pickled array)
│   ├── source_tpr                 -- intermediate data (pickled array)
│   ├── strategic_manipulation     -- intermediate data (pickled array)
│   ├── totals.csv               -- source data
│   └── upper_bound                -- intermediate data (pickled array)
```

# Figure 3b

Instructions:

Run replicator/main.py

```
├── replicator
│   ├── classifiers.py
│   ├── main.py
│   ├── readme.md
│   ├── responses.py
│   ├── setting.py
│   ├── s.p
│   ├── state.py
│   ├── system.py
│   ├── transfer.pdf
│   └── util.py
```

# Figures 1, 4, 9, 10

Instructions:

Run states/main.py

```
└── dp_states
    ├── FairTransferByState.py
    ├── FairTransferByYear.py
    └── main.py
```


# Figure 11

```
└── cov_states
│   ├── *_phi_g                    -- intermediate data (pickled array)
│   ├── *_phi_h                    -- intermediate data (pickled array)
│   ├── *_shifted_viol             -- intermediate data (pickled array)
│   ├── *_upper_bound              -- intermediate data (pickled array)
│   ├── *_cdf.pdf                -- source data per state
│   ├── *_default.pdf            -- source data per state
│   ├── totals.csv               -- source data
```
# Citation

If you want to cite our paper, please cite the following format:

```
@article{chen2022fairness,
  title={Fairness Transferability Subject to Bounded Distribution Shift},
  author={Chen, Yatong and Raab, Reilly and Wang, Jialu and Liu, Yang},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2022}
}
```
