# Raw Event Logs Directory

This directory stores original benchmark and domain event logs prior to RIMS+ preprocessing and filtering.

## Default Dataset: BPI Challenge 2012

The standard workflow uses the **BPI Challenge 2012** event log from a Dutch financial institute's personal loan and overdraft application process.

* **Expected File Path**: `data/raw/BPI_Challenge_2012.xes`
* **Source**: 4TU.ResearchData
* **DOI**: [10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)
* **Dataset Landing Page**: [https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204)

### Quick Download Command

You can download and extract the log into this folder using `curl`:

```bash
# From the project root:
curl -L -o data/raw/BPI_Challenge_2012.xes.gz "https://data.4tu.nl/file/3b3cb911-3701-42da-910b-5456f5eb478b/1"
gunzip data/raw/BPI_Challenge_2012.xes.gz
```

Once `BPI_Challenge_2012.xes` is in this folder, execute the pipeline starting with:

```bash
python src/00_filter_log.py
```

## Using Other Event Logs (Domain Agnostic)

To use other XES event logs (e.g. BAC, Sepsis, Road Traffic Fines, or custom internal logs):

1. Place your raw `.xes` or `.xes.gz` file in this directory (e.g. `data/raw/my_log.xes`).
2. Update the input path in [`config.yaml`](../../config.yaml):
   ```yaml
   preprocessing:
     input_log: "data/raw/my_log.xes"
   ```
3. Configure your activity and milestone prefixes (or set `process_activity_prefixes: []` and `milestone_prefixes: []` to model all activities with zero prefix filtering).
4. Run the pipeline starting with `python src/00_filter_log.py`.

