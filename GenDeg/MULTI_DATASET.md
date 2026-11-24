# Five-dataset composite training

The common loader supports CDD-11, VCD, CCDD-11, RFIR, and BID without
inventing unavailable physical labels. CDD-11 uses its native aligned-folder
loader. Each external dataset is converted once to JSONL and then sampled by
`CompositeDatasetMixture`.

## Experiment order

1. CDD-11: learn composite generation and type combinations.
2. VCD: add supervised intensity variation.
3. CCDD-11: add ratio, selective-control, and verified order labels.
4. RFIR: add natural-language and selective-degradation supervision.
5. BID: improve and evaluate real composite-weather generalization.

Do not train all five from scratch with equal sampling. Their labels and sizes
are different. Continue from the CDD-11 checkpoint and introduce each source
with a replay share of CDD-11 to avoid forgetting.

## JSONL contract

Every line minimally contains:

```json
{"clean":"clear/1.png","target":"haze_rain/1.png","active":["haze","rain"]}
```

Optional trusted labels are represented per degradation:

```json
{
  "clean": "clear/1.png",
  "target": "haze_rain/1.png",
  "active": ["haze", "rain"],
  "prompt": "an outdoor scene with moderate haze and heavy rain",
  "factors": {
    "haze": {"stats": [0.12, 0.04], "parameters": [0.7, 0.2],
             "parameter_mask": [1, 1], "order": 0},
    "rain": {"parameters": [0.8, 0.35],
             "parameter_mask": [1, 1], "order": 1}
  }
}
```

Unknown parameters are omitted. The loader leaves their values at zero and
their masks at zero, so zero is never confused with a measured physical value.
Folder-name order is not accepted as verified supervision.

`configs/train_composite_5dataset_template.yaml` is intentionally inactive
until all referenced manifests and data roots exist.
