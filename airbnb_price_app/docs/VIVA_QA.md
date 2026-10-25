# Viva Questions and Answers

## Web scraping
**Q1. What does `requests.get()` return and what do status codes mean?** A `Response` object (`.status_code`, `.text`, `.headers`). 200 = OK, 404 = not found, 403 = forbidden, 429 = too many requests, 5xx = server error.
**Q2. `find()` vs `find_all()`?** `find` returns the first matching tag (or `None`), `find_all` returns a list of all matches. `get_text()` extracts the text; `tag["href"]` reads an attribute.
**Q3. How do you scrape responsibly?** Check robots.txt and the terms of service, identify yourself with a User-Agent, add delays, handle errors/retries, avoid personal data. Airbnb's terms forbid scraping, hence the offline demo and Inside Airbnb CSVs.

## Cleaning & descriptive statistics
**Q4. How were missing values handled?** Explored with `isna()`; for modelling, median (numeric) / most-frequent (categorical) imputation *inside the Pipeline*, fitted on training data only.
**Q5. What is the IQR rule?** Outliers lie outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] where IQR = Q3 − Q1. Option to remove or cap (winsorise).
**Q6. Why is mean > median for prices?** Prices are right-skewed: a few very expensive listings pull the mean up; median is robust.
**Q7. Variance vs standard deviation?** SD = √variance, in the original units, so easier to interpret.

## Probability & CLT
**Q8. Population vs sample; what is a random variable?** Population = all listings; sample = observed subset; a random variable maps outcomes to numbers (e.g. price).
**Q9. When do you use Binomial vs Poisson?** Binomial: fixed number *n* of independent yes/no trials; Poisson: count of events per interval with rate λ (mean = variance = λ).
**Q10. State the CLT.** For i.i.d. samples with finite variance, the distribution of the sample mean approaches Normal(μ, σ²/n) as n grows, regardless of the population's shape. Demonstrated with skewed prices.
**Q11. Standard error vs standard deviation?** SD describes spread of individual values; SE = s/√n describes spread of the *sample mean*.
**Q12. What does a 95 % CI mean?** If we repeated the sampling many times, about 95 % of such intervals would contain the true mean (shown by the coverage simulation). It is not "95 % probability the mean is in this interval".
**Q13. Why are prices modelled as log-normal?** Multiplicative effects (bedrooms, location) → log(price) is approximately Normal; the KS test and Q–Q plot support it.

## Hypothesis testing & correlation
**Q14. Define null/alternative, α, p-value, Type I/II error.** H0 = no effect; H1 = effect; α = tolerated Type-I error rate (rejecting a true H0); p = P(data at least this extreme | H0); Type II = failing to reject a false H0.
**Q15. Student vs Welch t-test?** Student assumes equal variances; Welch does not and is the safer default. Levene's test checks equal variances.
**Q16. Paired vs independent t-test?** Paired: the same listing measured twice (weekday vs weekend price) → test the mean of differences. Independent: different listings.
**Q17. When ANOVA, and why post-hoc?** Compare ≥ 3 group means; a significant F only says *some* mean differs; Tukey HSD finds which pairs while controlling the family-wise error.
**Q18. Chi-square test of independence?** Tests association between two categorical variables using observed vs expected counts; expected counts should be ≥ 5; Cramér's V gives the strength.
**Q19. Covariance vs correlation?** Covariance is scale-dependent; correlation = covariance/(σxσy) lies in [−1, 1].
**Q20. Pearson vs Spearman?** Pearson = linear association, sensitive to outliers; Spearman = rank-based, monotonic association, robust.
**Q21. Does correlation imply causation?** No — confounders (e.g. size drives both bedrooms and price), reverse causality, coincidence.
**Q22. Statistical vs practical significance?** With thousands of rows tiny differences become "significant"; report effect size (Cohen's d, η², Cramér's V).

## Regression & evaluation
**Q23. Linear regression assumptions?** Linearity, independent errors, constant variance (homoscedasticity), roughly Normal residuals, no severe multicollinearity.
**Q24. Ridge vs Lasso?** Ridge adds λΣβ² (shrinks coefficients); Lasso adds λΣ|β| (can set coefficients exactly to 0 → feature selection).
**Q25. Decision tree vs random forest?** A tree splits on features recursively and over-fits easily; a forest averages many trees on bootstrap samples with random feature subsets → lower variance.
**Q26. Why one-hot encode and scale?** Models need numbers; one-hot avoids implying an order among categories. Scaling matters for Ridge/Lasso; Min-Max → [0,1]; StandardScaler → mean 0, SD 1. Trees don't need it.
**Q27. Define MAE, MSE, RMSE, R².** MAE = mean |error|; MSE = mean error²; RMSE = √MSE (same units as price, penalises big errors); R² = 1 − SSres/SStot (share of variance explained).
**Q28. What is data leakage and how was it avoided?** Information from the test set (or the target) influencing training. Avoided by splitting first, fitting imputer/scaler/encoder inside the Pipeline, tuning with CV on training data, excluding weekday/weekend price columns, and choosing the model by CV not test score.
**Q29. Why predict log(price)?** Reduces skew, stabilises variance, matches multiplicative effects; predictions are back-transformed (`expm1`) so metrics are in dollars.
**Q30. Why might a linear model beat a random forest here?** The synthetic data is log-linear by construction; on real data non-linearities/interactions usually favour forests. Model choice must be empirical.
**Q31. What do residual plots reveal?** Patterns (non-linearity), funnels (heteroscedasticity), skew/heavy tails (outliers).
**Q32. Limitations of the project?** Synthetic data, no amenities/photos/season, point estimates without intervals, not an Airbnb quote.
