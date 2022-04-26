# %%
from statsmodels.stats.proportion import proportions_ztest

# %%
# sec4.1
""". The proportion of the participants whoanswer Test18 correctly in Phase 2 (39%) is significantly higher thanthe proportions in Phase 1 (24%) and in Phase 3 (21%"""
p1_total, p2_total, p3_total = 85, 99, 105
p1_correct, p2_correct, p3_correct = round(0.24*p1_total), round(0.39*p2_total), round(0.21 *p3_total)
print(proportions_ztest([p2_correct, p1_correct], [p2_total, p1_total]))
print(proportions_ztest([p2_correct, p3_correct], [p2_total, p3_total]))
