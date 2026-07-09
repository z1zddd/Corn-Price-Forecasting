# Model Families

Model families group operators by modeling method:

- `baseline/`: simple reference models.
- `classical/`: tabular sklearn-style models and optional tree boosters.
- `sequence/`: optional torch-backed sequence models.
- `official/`: the 57-model pool split into `tabular/`, `aeon/`, and `keras/`.
- `aggregation/`: operators that combine existing prediction streams.

No evaluation pipeline, report writer, experiment output, or smoke script should
live here.
