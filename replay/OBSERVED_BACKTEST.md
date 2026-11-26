# Observed-outcome benchmark

Measured gas consumption in the years before and after loft insulation, excluding overlapping recorded cavity-wall and PV installations and requiring valid meter flags. Both methods use the pre-installation annual gas reading. The pooled median and property-type medians are fitted through installation year 2015 and scored on 2018–2023 installations.

| Scored cases | Baseline mean loss | Candidate mean loss | Candidate wins | Baseline wins | Ties |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 3716.343750 | 3739.370174 | 31 | 33 | 0 |

**Metric:** MAE in kWh/year. Lower is better. See [aggregate results](observed-results.json) for the largest regression and breakdowns.

The pooled baseline was slightly better. This evaluates observed changes in consumption. Weather, occupancy, energy prices and other changes can affect consumption, so the pilot does not establish causal retrofit savings or validate household bills.

## Reproduce

Save [DESNZ anonymised NEED sample](https://assets.publishing.service.gov.uk/media/6a282ad83b15d05a7ce31fb1/anon2026_50k.csv) as `need.csv`. The input file SHA-256 is `5da6b2c1d6c2296f200a3ff3f8e3e05d2141fd40767e51cd5f3c57faa501b50b`. The workbook requires `xlrd`; all benchmarks use `pandas` and `numpy`.

```sh
python replay/observed_backtest.py need.csv --output /tmp/observed-results.json
diff -u replay/observed-results.json /tmp/observed-results.json
```
