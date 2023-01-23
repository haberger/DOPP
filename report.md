# Corruption
## Group 21: Godun, Haberger, Haberl, Konzett

To answer the following question on the topic of corruption a model trained on the [QoG (Quality of Governence) Standard Time-Series Dataset](https://www.gu.se/en/quality-government/qog-data/data-downloads/standard-dataset) was created and analyzed: 

- Which country’s characteristics are good predictors of a country’s position on a corruption index?

- Are there countries whose characteristics don’t correspond to their position on a corruption index?

- If yes, does this hold across all corruption indexes, or do the lists of such countries vary from index to index?

### Characteristics

![](img/feature.png)

In the plot above the most important country characteristics for corruption prediction are shown in decreasing order of importance.
The most important features can be categorized in the following way:
- __political freedom__: "fh_pr_1.0", "fh_ipolity2", "br_mon", "cpds_vper_0.0", "wel_sys_1.0", "fh_status_1.0", 
- __Life expectency__: "ihme_lifexp_0104t", ihme_lifexp_0104m
- __civil liberties__: "gd_ptss_1.0", "fh_cl_1.0 "
