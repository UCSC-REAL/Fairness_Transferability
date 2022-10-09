# Fair Transferability Subject to Bounded Distribution Shift

(Neurips 2022)

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

@article{chen2022fairness,
  title={Fairness Transferability Subject to Bounded Distribution Shift},
  author={Chen, Yatong and Raab, Reilly and Wang, Jialu and Liu, Yang},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2022}
}
