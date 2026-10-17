Week 3: Statistical Analysis and Hypothesis Testing (Health Insurance Costs)
Virtual Data Science with Python Apprenticeship, Week 3 task.

Overview
A hypothesis-testing study of what drives annual medical insurance charges, using the public medical insurance cost dataset (1,337 customers after removing one duplicate).

Hypotheses and tests
#	Hypothesis	Test	Result
H1	Smokers incur higher charges	Welch t-test, Mann-Whitney, bootstrap	Supported (d = 3.16)
H2	Obese customers incur higher charges	Welch t-test, stratified by smoking	Supported, but concentrated in smokers
H3	Charges differ by region	ANOVA, Kruskal-Wallis, Tukey HSD	Not supported (fails robustness checks)
H4	Smoking status is related to sex	Chi-square, odds ratio	Supported, but weak (V = 0.08)
H5	Age and charges are correlated	Pearson, Spearman	Supported (r = 0.30; 0.63 in non-smokers)
H6	Combined effect of all factors	OLS regression on log(charges)	R² = 0.77, smoking dominates
Assumption checks (Shapiro-Wilk, Levene, Q-Q plots), effect sizes, 95% confidence intervals and a Bonferroni check for multiple testing are included.

Files
week3_stats_analysis.py: script that runs every test and creates all figures
insurance_clean.csv: cleaned dataset
Week3_Hypothesis_Testing_Report.docx: full written report
How to run
pip install pandas numpy scipy statsmodels matplotlib seaborn
python week3_stats_analysis.py
