# Raw Corn Materials

Raw materials are source-like inputs before framework modeling transforms.
Examples include original corn price exports, futures tables, spot price
sources, weather extracts, or vendor/source snapshots.

Current tracked raw fixture:

- `玉米价格原始数据.csv`: source-like corn price/factor export used as a small reproducible fixture.

Keep large raw data, private enterprise feeds, credentials, and archives outside
git, for example under `local_data/`.
