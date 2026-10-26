# Data

The call transcripts are not included in this repository. They are derived from the
Korean voice phishing dataset of Cho and Seo (NAACL 2025 Industry Track), which is
distributed on request. Place the following files here, or point `SCL_DATA_ROOT` to the
folder that contains them:

```
data/
├── gold_final.csv                  # 3,014 relabelled transcripts (the search corpus)
├── finance_ood_challenge.csv       # 224 finance-impersonation queries
└── government_ood_challenge.csv    # 447 government-impersonation queries
```

| File | Required columns | Notes |
|---|---|---|
| `gold_final.csv` | `review_id`, `gold_label`, `text` | `gold_label` is `finance` (2,100) or `government` (914) |
| `*_ood_challenge.csv` | `text`, `category` | every row has the category in the file name |

All files are UTF-8. The row order matters: document and query ids are assigned in
file order (`gold_<category>_<5-digit number>` per category, `<direction>_q<4-digit
number>`), and the released qrels in `../qrels/` refer to those ids.
