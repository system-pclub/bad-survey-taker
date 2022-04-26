# %%
# todo: max string length
# browser version for classification
# 2 sec enforced 

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import re, json, math
import random
import sqlite3
import pingouin as pg
from pprint import pprint
from collections import Counter
from IPython.display import clear_output, display, Markdown as md
from scipy import stats


# %% [markdown]
# ## Functions

# %%
from lib import *

# %%


# %% [markdown]
# ## Load Data and Process Data

# %%
if ANALYZE_RUST:
    trajectory_data = pd.read_xml(trajectory_data_path)
    trajectory_data = trajectory_data.rename(columns={"_recordId": "ResponseId"})[["ResponseId", "mouseTrajectory"]]
    unfinished_data = pd.read_excel(rust_unfinished_qualtrics_path, "incomplete(remove no Q31)", skiprows=[1])
    finished_data = pd.read_excel(rust_qualtrics_path, "complete", skiprows=[1])
    recode_data = pd.read_excel(rust_qualtrics_recode_path, skiprows=[1])
    finished_data = merge_raw_data(finished_data, recode_data)
    finished_data = finished_data.merge(trajectory_data, on="ResponseId")
    query_db_path = rust_query_db_path
    
else:
    unfinished_data = pd.read_csv(deepfake_unfinished_qualtrics_path, skiprows=[1,2])
    finished_data = pd.read_csv(deepfake_qualtrics_path)
    finished_data = finished_data.rename(columns={'Approve':'valid?'})
    finished_data = finished_data.rename(columns={'id':'uid'})

    display(finished_data)
    finished_data['valid?'] = finished_data['valid?'].apply(lambda x: 1 if x=='x' else 0)


unfinished_data = unfinished_data[unfinished_data["Finished"] == False]
finished_data = finished_data[finished_data["Finished"] == True]



if ANALYZE_RUST: #other study doesn't have this yet
    qid = "QID42"
    finished_data["mouseTrajectory"] = finished_data["mouseTrajectory"].apply(lambda x: parse_qid_json(parse_qid_data_entry(x)))
    finished_data["mouseTrajectory_max_speed"] = finished_data["mouseTrajectory"].apply(lambda x: find_trajectory_max_speed(x.get(qid)))
    finished_data["mouseTrajectory_sequence_length"] = finished_data["mouseTrajectory"].apply(lambda x: find_trajectory_sequence_length(x.get(qid)))
    finished_data["mouseTrajectory_path"] = finished_data["mouseTrajectory"].apply(lambda x: find_trajectory_path(x.get(qid)))
    finished_data[finished_data["ResponseId"] == "R_3sAHkpNM0AtsFuc"]["mouseTrajectory"].apply(lambda x: x.get(qid))

display(unfinished_data)
display(finished_data)


# %%
DATA_GROUP = "finished"
# DATA_GROUP = "unfinished"


if DATA_GROUP == "finished":
    data = finished_data
    
else:
    data = unfinished_data

data["valid?"] = data["valid?"].fillna(0).astype("int32")
#new columns
if ANALYZE_RUST:
    data['consistent_answers'] = data.apply(rust_consistent_answers, axis=1)
    data['passed_attention'] = data.apply(rust_attention_correct, axis=1)
else:
    data['consistent_answers'] = data.apply(consistent_answers, axis=1)
    data['passed_attention'] = data.apply(attention_correct, axis=1)

valid_data = data[data["valid?"] == 1]
invalid_data = data[data["valid?"] != 1]

prefix = DATA_GROUP + "_"

# %% [markdown]
# # Unfinished Data Analysis

# %%
unfinished_data["Progress"].plot.hist(title="Progress of Unfinished Responses")

# %% [markdown]
# # Finished Data Analysis

# %% [markdown]
# ### Timing Analysis

# %%
plt.clf()
ax = sns.histplot(pd.DataFrame({
    "valid": (valid_data["Duration (in seconds)"] / 60)[lambda x: x < 120], 
    "invalid": (invalid_data["Duration (in seconds)"] / 60)[lambda x: x<120], 
}), stat="probability", bins=40, element="step", common_norm=False)
ax.set_xlabel("Duration (minute)")

ax = pd.DataFrame({
    "valid": (valid_data["Duration (in seconds)"] / 60)[lambda x: x >= 120].shape, 
    "invalid": (invalid_data["Duration (in seconds)"] / 60)[lambda x: x>=120].shape, 
})
display(ax)


# %% [markdown]
# ### Classifier Evaluation

# %%
method_category = {
    "UID missing": "Redirect Page", 
    "Unexpected referrer": "Redirect Page",
    "Timezone mismatch": "VPN", 
    "WebRTC": "VPN",
    "VirusToal (IP)": "Activity History", 
    "MinFraud (IP)": "Activity History",
    "IPRegistry (IP)": "Activity History",
    "DNSBL (IP)": "Activity History", 
    "RelevantID (Fraud)": "Activity History", 
    "UID": "Duplication", 
    "Cookie": "Duplication", 
    "IPAddress": "Duplication", 
    "Browser Fingerprint": "Duplication", 
    "RelevantID (Duplicate)": "Duplication", 
    "reCAPTCHA": "Automation", 
    "Waiting for 5 seconds": "Automation", 
    "Resolution": "Automation", 
    "Attention Check": "Psychology", 
    "Consistency Check": "Psychology", 
    "Techniques" : "Ensemble", 
    "RelevantID": "Ensemble", 
    "Rust": "Ensemble"
}

# %%
data["Q_RelevantIDDuplicate"] = data["Q_RelevantIDDuplicate"].fillna(False)
method_eval_df = pd.DataFrame(columns = ["Method", "Category", "TP", "FP", 
                                         "TN", "FN", 
                                         "valid missing", "invalid missing", 
                                         "precision", "recall", "f-score"])

result = element_is_blank(data, "UID missing", 'uid')
method_eval_df = method_eval_df.append(result, ignore_index=True)

method_eval_df = eval_column_method(method_eval_df, "Unexpected referrer", data, "Referer", 
                                    lambda x: expected_referer not in x if type(x) == str else True, 
                                    is_missing=lambda x: False)

result = compare_os_timezone_with_ip(data, query_db_path)
method_eval_df = method_eval_df.append(result, ignore_index=True)

result = compare_webrtc_ip_missing_as_invalid(data)
method_eval_df = method_eval_df.append(result, ignore_index=True)

predict_with_ip_databases(data, query_db_path)
method_eval_df = eval_column_method(method_eval_df, "VirusToal (IP)", data, "VirusTotal_predict", lambda x: x, pd.isna)
method_eval_df = eval_column_method(method_eval_df, "MinFraud (IP)", data, "MinFraud_predict", lambda x: x, pd.isna)
method_eval_df = eval_column_method(method_eval_df, "IPRegistry (IP)", data, "IPRegistry_predict", lambda x: x, pd.isna)
method_eval_df = eval_column_method(method_eval_df, "DNSBL (IP)", data, "DNSBL_predict", lambda x: x, pd.isna)

method_eval_df = eval_column_method(method_eval_df, "RelevantID (Fraud)", data, 
                                    "Q_RelevantIDFraudScore", lambda x: x >= 30, is_missing=lambda x: False)

result = elements_has_duplication(data, "UID", "uid")
method_eval_df = method_eval_df.append(result, ignore_index=True)

#jaron: this function doesn't work for me?
result = compare_cookie(data)
method_eval_df = method_eval_df.append(result, ignore_index=True)

result = elements_has_duplication(data, "IPAddress","IPAddress")
method_eval_df = method_eval_df.append(result, ignore_index=True)

result = qid_data_find_duplicate(data, "Browser Fingerprint", "browserFingerprint")
method_eval_df = method_eval_df.append(result, ignore_index=True)

method_eval_df = eval_column_method(method_eval_df, "RelevantID (Duplicate)", data, 
                                    "Q_RelevantIDDuplicate", lambda x: x, is_missing=lambda x: False)

#reCAPTCHA, manually calculated
if ANALYZE_RUST:
    tp = 8
    fp = 2
    tn = 50
    fn = 239  
    valid_missing = 14
    invalid_missing = 147
else:
    tp = 0
    fp = 0
    tn = 0 + len(valid_data.index)
    fn = 0 + len(invalid_data.index)
    valid_missing = 7
    invalid_missing = 8
result = pd.Series({"Method": "reCAPTCHA", "TP": tp, "FP": fp, 
                        "TN": tn, "FN": fn, 
                        "valid missing": valid_missing, "invalid missing":invalid_missing, 
                        "precision": calculate_precision(tp, fp, tn, fn), "recall": calculate_recall(tp, fp, tn, fn),
                        "f-score" : calculate_f_score(tp, fp, tn, fn)
                       })
method_eval_df = method_eval_df.append(result, ignore_index=True)


#TODO ZIYI: add results for this
#Waiting an interval, manually calculated
if ANALYZE_RUST:
    tp = 1
    fp = 0
    tn = 52
    fn = 247  
    valid_missing = 14
    invalid_missing = 146
else: 
    tp = 3
    fp = 2
    tn = 0 + len(valid_data.index)
    fn = 0 + len(invalid_data.index)
    valid_missing = 5
    invalid_missing = 5
result = pd.Series({"Method": "Waiting for 5 seconds", "TP": tp, "FP": fp, 
                        "TN": tn, "FN": fn, 
                        "valid missing": valid_missing, "invalid missing":invalid_missing, 
                        "precision": calculate_precision(tp, fp, tn, fn), "recall": calculate_recall(tp, fp, tn, fn),
                        "f-score" : calculate_f_score(tp, fp, tn, fn)
                       })
method_eval_df = method_eval_df.append(result, ignore_index=True)

result = evaluate_odd_resolution(data)
method_eval_df = method_eval_df.append(result, ignore_index=True)

method_eval_df = eval_column_method(method_eval_df, "Attention Check", data, "passed_attention", 
                                    lambda x: not x, is_missing=pd.isna)
method_eval_df = eval_column_method(method_eval_df, "Consistency Check", data, "consistent_answers", 
                                    lambda x: not x, is_missing=pd.isna)

result = predict_on_trajectory_path(data, "mouseTrajectory_path")
method_eval_df = method_eval_df.append(result, ignore_index=True)
# result = compare_os_timezone_with_ip(data, query_db_path)
# method_eval_df = method_eval_df.append(result, ignore_index=True)

# result = compare_webrtc_ip(data)
# method_eval_df = method_eval_df.append(result, ignore_index=True)

# result = mouse_moved_evaluation(data, "mouseMoved")
method_eval_df = method_eval_df.append(result, ignore_index=True)

if ANALYZE_RUST:
    data['knowledge_1_correct'] = data.apply(rust_knowledge1_correct, axis=1)
    data['knowledge_2_correct'] = data.apply(rust_knowledge2_correct, axis=1)
    data["knowledge_1&2_correct"] = data['knowledge_1_correct'] & data['knowledge_2_correct']
    method_eval_df = eval_column_method(method_eval_df, "Knowledge Check 1", data, "knowledge_1_correct", 
                                    lambda x: not x, is_missing=pd.isna)

    method_eval_df = eval_column_method(method_eval_df, "Knowledge Check 2", data, "knowledge_2_correct", 
                                    lambda x: not x, is_missing=pd.isna)
    method_eval_df = eval_column_method(method_eval_df, "Knowledge Check 1&2", data, "knowledge_1&2_correct", 
                                    lambda x: not x, is_missing=pd.isna)

# result = (data["Unexpected referrer_predict"] | data["UID missing_predict"] | data["UID_predict"] | data["Cookie_predict"] 
#   | data["IPAddress_predict"] | data["Browser Fingerprint_predict"] 
#   | data["Resolution_predict"])
technical_ensemble = ["Unexpected referrer_predict", "UID missing_predict", "UID_predict", "Cookie_predict","IPAddress_predict", "Browser Fingerprint_predict","Resolution_predict"]
result = kleene_or_columns(data, technical_ensemble)
data["technical_ensemble (Test24)"] = result
method_eval_df = eval_column_method(method_eval_df, "Techniques (test 24)", data, "technical_ensemble (Test24)", 
                                    lambda x: x, is_missing=pd.isna)

relevantID_ensemble = ["RelevantID (Fraud)_predict", "RelevantID (Duplicate)_predict"]
result = kleene_or_columns(data, relevantID_ensemble)
data["relevantID_ensemble"] = result 
method_eval_df = eval_column_method(method_eval_df, "RelevantID", data, "relevantID_ensemble", 
                                    lambda x: x, is_missing=pd.isna)
# 4,6,13,18,19
technical_ensemble = ["WebRTC_predict", "MinFraud (IP)_predict", "Browser Fingerprint_predict", "Attention Check_predict", "Consistency Check_predict"]
result = kleene_or_columns(data, technical_ensemble)
display(result)
data["technical_ensemble (Test 4,6,13,18,19)"] = result
method_eval_df = eval_column_method(method_eval_df, "Techniques (Test 4,6,13,18,19)", data, "technical_ensemble (Test 4,6,13,18,19)", 
                                    lambda x: x, is_missing=pd.isna)

# 18,19
technical_ensemble = ["Attention Check_predict", "Consistency Check_predict"]
result = kleene_or_columns(data, technical_ensemble)
display(result)
data["technical_ensemble (Test 18,19)"] = result
method_eval_df = eval_column_method(method_eval_df, "Techniques (Test 18,19)", data, "technical_ensemble (Test 18,19)", 
                                    lambda x: x, is_missing=pd.isna)

# 4,13,19
technical_ensemble = ["WebRTC_predict", "Browser Fingerprint_predict", "Consistency Check_predict"]
result = kleene_or_columns(data, technical_ensemble)
display(result)
data["technical_ensemble (Test 4,13,19)"] = result
method_eval_df = eval_column_method(method_eval_df, "Techniques (Test 4,13,19)", data, "technical_ensemble (Test 4,13,19)", 
                                    lambda x: x, is_missing=pd.isna)
                                    
method_eval_df["Category"] = method_eval_df["Method"].apply(lambda x: method_category.get(x) or "")
display(method_eval_df)

# %%


# %%
def calculate_combinations_of_two(method_eval_df, data, and_logic=True):
    methods = list(data.filter(axis="columns", like="_predict").columns)
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            if and_logic: 
                combination = f'2-AND:{methods[i]} & {methods[j]}'
                print(combination)
                data[combination] = data[methods[i]] & data[methods[j]]
            else:
                combination = f'2-OR:{methods[i]} & {methods[j]}'

                print(combination)
                print(data[methods[i]].dtypes, data[methods[j]].dtypes)


                data[combination] = data[methods[i]] | data[methods[j]]
            method_eval_df = eval_column_method(method_eval_df, combination, data, combination, 
                                    lambda x: x, is_missing=lambda x: False)
    return method_eval_df
            
def calculate_combinations_of_three(method_eval_df, data, and_logic=True):
    methods = list(data.filter(axis="columns", like="_predict").columns)
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            for k in range(j+1, len(methods)):
                if and_logic:
                    combination = f'3-AND:{methods[i]} & {methods[j]} & {methods[k]}'
                    data[combination] = data[methods[i]] & data[methods[j]] & data[methods[k]]
                else:
                    combination = f'3-OR:{methods[i]} & {methods[j]} & {methods[k]}'
                    data[combination] = data[methods[i]] | data[methods[j]] | data[methods[k]]
                method_eval_df = eval_column_method(method_eval_df, combination, data, combination, 
                                    lambda x: x, is_missing=lambda x: False)
                print(combination)
    return method_eval_df

#method_eval_df_two_or = pd.DataFrame(columns = ["Method","TP", "FP", "TN", "FN", "valid missing", "invalid missing", "precision", "recall" ])
#method_eval_df_two_or = calculate_combinations_of_two(method_eval_df_two_or, data, and_logic=False)

#method_eval_df_three_or = pd.DataFrame(columns = ["Method","TP", "FP", "TN", "FN", "valid missing", "invalid missing", "precision", "recall" ])
#method_eval_df_three_or = calculate_combinations_of_three(method_eval_df_three_or, data, and_logic=False)

#method_eval_df_two_and = pd.DataFrame(columns = ["Method","TP", "FP", "TN", "FN", "valid missing", "invalid missing", "precision", "recall" ])
#method_eval_df_two_and = calculate_combinations_of_two(method_eval_df_two_and, data )

#method_eval_df_three_and = pd.DataFrame(columns = ["Method","TP", "FP", "TN", "FN", "valid missing", "invalid missing", "precision", "recall" ])
#method_eval_df_three_and = calculate_combinations_of_three(method_eval_df_three_and, data)


# %%
def plot_methods(title, method_eval_df):
    #graph results
    if ANALYZE_RUST:
        plot_title = "Detection Effectiveness on Rust Survey"
    else:
        plot_title = 'Detection Effectiveness on Mturk'

    display(method_eval_df)
    x = method_eval_df['recall'].values #league_rushing_success['success'].values
    y = method_eval_df['precision'].values #league_rushing_success['epa'].values
    types = method_eval_df['Method'].values
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x, y)

    ax.set_xlabel('Recall', fontsize=20)
    ax.set_ylabel('Precision', fontsize=20)

    ax.set_title(plot_title, fontsize=22)
    for i,row in method_eval_df.iterrows():
        ax.annotate(i, (x[i], y[i]), xytext=(10,random.randrange(4,14)), textcoords='offset points')
        plt.scatter(x, y, marker='x', color='red')
        plt.axis([-0.1, 1.1, -0.1, 1.1])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.hlines(y=0.5,xmin=0.0, xmax=1.0, color='b', linestyle='--')
        plt.text(0.5, 0.48, 'baseline classifier', fontsize=12, color='grey',va='center', ha='center')
    plt.show()
    fig.savefig("mturk-precision-recall.pdf")
        
if ANALYZE_RUST:
    title = "Detection Effectiveness on Rust Survey"
else:
    title = "Detection Effectiveness on Mturk"
    
plot_methods(title, method_eval_df)
#plot_methods(title, method_eval_df_two_or)
#plot_methods(title, method_eval_df_three_or)
#plot_methods(title, method_eval_df_two_and)
#plot_methods(title, method_eval_df_three_and)

# %%
# invalid = data[data["valid?"] != 1]
# invalid[(invalid["WebRTC-different_predict"] == True) & (invalid["Unexpected referrer_predict"] == False)]

# %%
#ttest_valid_invalid_distribution(data, "Q8_Page Submit")

# %%
def print_latex_table(eval_method_df, study_name):
    latex_md_header = """
\\begin{table*}[!htp]\centering
\\small
\\begin{tabular}{lll|cccccc}\\toprule
ID & Description                & Category &TP &FP &TN &FN     &Precision &Recall \\\\\\\\ 
\\midrule
    """
    latex_md_footer = f"""
\\bottomrule  
\\end{{tabular}}  
\\caption{{ \\textbf{{{study_name} Method Evaluation}}— \\textmd{{\\small texttexttext.}} }}   
\\label{{table:mturk-method-eval}} \\vspace{{-0.12in}}
\\end{{table*}}
    """
    def escape_markdown(text):
        # Use {} and reverse markdown carefully.
        parse = re.sub(r"([_*\[\]()~`>\#\+\-=|\.!])", r"\\\1", text)
        reparse = re.sub(r"\\\\([_*\[\]()~`>\#\+\-=|\.!])", r"\1", parse)
        return reparse 

    table_content = ""
    for i, row in eval_method_df.iterrows():
        table_row_str = f"Test{i+1} & {escape_markdown(row['Method'])} & {escape_markdown(row['Category'])} " \
        f" & {round(row['TP'])}  & {round(row['FP'])}" \
        f" & {round(row['TN'])}  & {round(row['FN'])}" \
        f" & {row['precision']:.3f}  & {row['recall']:.3f} \\\\\\\\"
        table_content += table_row_str + "  \n"
        
    #display(md(latex_md_header))
    display(md(latex_md_header + table_content + latex_md_footer ))
if ANALYZE_RUST:
    study_name = "Rust"
else:
    study_name = "MTurk"

print_latex_table(method_eval_df, study_name)

# %%
method_eval_df.to_clipboard(excel=True)

# %% [markdown]
# ## Stats

# %%
# p1
ttest_valid_invalid_distribution(data, "Q23_1")
# p2
ttest_valid_invalid_distribution(data, "Q188_1")


# %%

# p2 Q75, Q91
res = mixed_anova_test(data, withins=["Q75_1", "Q91_1"], valid_col="valid?", within_labels=["base", "enhanced"])
res.to_clipboard(excel=True)
res

# %%
res = mixed_anova_test(data, withins=["Q82_1", "Q98_1"], valid_col="valid?", within_labels=["base", "enhanced"])
res.to_clipboard(excel=True)
res

# %%
show_os_timezone_comparison(data)

# %%
valid = data[data["valid?"] == 1]
valid[valid["consistent_answers"] == False][["Q20", "Q191"]]

# %%
invalid = data[data["valid?"] != 1]
bot = data[data["Resolution_predict"]]
count_browser_versions(bot)


# %%
count_browser_versions(invalid)

# %%
cols = ["Q13", "Q49", "Q193"]
for i, col in enumerate(cols):
    print(f"Phase {i+1}")
    valid = data[data["valid?"] == 1][col].dropna()

    print("valid correct count:", (valid == 2).sum(), "correct proportion:", (valid == 2).sum() / len(valid))
    invalid = data[data["valid?"] != 1][col].dropna()

    print("invalid correct count:", (invalid == 2).sum(), "correct proportion:", (invalid == 2).sum() / len(invalid))


# %%

# plot_all_click_counts(data, prefix)
calculate_phase_RT(data, phase_questions)


# %%
res, fdata = show_mean_error(data, "Duration (in seconds)")
res

# %%
fdata.shape

# %%
(data["Q193"] == 2).sum() / (len(data["Q193"].dropna()))

# %%
ttest_valid_invalid_distribution(fdata, "Phase 1_1 RT")

# %%
mwu_valid_invalid_distribution(fdata, "Duration (in seconds)")

# %%
mwu_valid_invalid_distribution(fdata, "Phase 1_1 RT")

# %%
mwu_valid_invalid_distribution(fdata, "Phase 3 RT")

# %%
pg.normality(fdata[fdata["valid?"] == 1]["Duration (in seconds)"])

# %%
(data["Duration (in seconds)"] > 7200).sum()
# %%
fdata[(fdata["uid"] == "e0528e46282f4651833a9e22b02a4e6c") & (fdata["Consistency Check_predict"] == True)].shape

# %%
data[(data["uid"] == "e0528e46282f4651833a9e22b02a4e6c") & (data["Consistency Check_predict"] == True)].shape

# %%
data[(data["uid"] == "e0528e46282f4651833a9e22b02a4e6c") & (data["Attention Check_predict"] == True)].shape

# %%
case_data = data[data["uid"] == "e0528e46282f4651833a9e22b02a4e6c"]
case_data[["Q13", "Q49", "Q193"]].fillna(0).sum(axis=1).value_counts()

# %%
# set(data["webrtcIP"].apply(parse_qid_data_entry).apply(lambda x: [] if not x else list(x.values())[0].split(", ")).apply(lambda x: x[-1] if len(x)>0 else ""))


# %%
strange = data[data[f"{fingerprint_q}_Resolution"] == "1364x615"]
plot_one_question_choices(strange, "Q98_1", "strange_")



# %%
# mixed_anova_test(data, withins=[["Q77_1", "Q78_1", "Q79_1", "Q80_1", "Q81_1", "Q82_1"],
#     ["Q93_1", "Q94_1", "Q95_1", "Q96_1", "Q97_1", "Q98_1"]], between_col="valid?")

# %%
# valid_res, nb_invalid_res, bot_res = compute_response_RT_skew_kurtosis(data)
valid_res, nb_invalid_res, bot_res = compute_response_RT_skew_kurtosis(fdata, lambda x: x["Consistency Check_predict"])
valid_res.to_clipboard(excel=True)

# %%
# appendix table 2

res, _ = show_mean_error(fdata, "Phase 1_1 RT", False)
# print("1.1 RT", res)


res, _ = show_mean_error(fdata, "Phase 1_2 RT", False)
# print("1.2 RT", res)

res, _ = show_mean_error(fdata, "Phase 2 RT", False)
# print("2 RT", res)

res, _ = show_mean_error(fdata, "Phase 3 RT", False)
# print("3 RT", res)

res, _ = show_mean_error(fdata, "Duration (in seconds)", False)
res

# Q3 (wait 2s page), Q196 (wait 5s)
res, _ = show_mean_error(fdata, "Q3_Page Submit", False)

show_mean_error(fdata, "Q196_Page Submit", False)

# captcha RT
show_mean_error(fdata, "Q198_Page Submit", False)

# %%

data[(data["Attention Check_predict"] == True) & (data["valid?"] == 1)][["Q13", "Q49", "Q193"]].fillna(0).sum(axis=1).value_counts()

# %%
#appendix table 3
questions = {
    "rust programming experience": "Q23_1",
    "all programming experience": "Q188_1",
    "difficulty before err msg": "Q61_1",
    "difficulty base msg": "Q75_1",
    "difficulty enhanced msg": "Q91_1"
}

for name, qid in questions.items():
    print(name)
    show_mean_error(data, qid, False)

# %%

res = mixed_anova_test(data, 
    withins=["Q61_1", "Q75_1", "Q91_1"], valid_col="valid?", within_labels=["before", "base", "enhanced"])
display(res)

# %%
# table4
entries = {"Mental": ["Q77_1", "Q93_1"], 
    "Physical": ["Q78_1", "Q94_1"], 
    "Temporal": ["Q79_1", "Q95_1"], 
    "Performance": ["Q80_1", "Q96_1"], 
    "Effort": ["Q81_1", "Q97_1"], 
    "Frustration": ["Q82_1", "Q98_1"]
    }

for name, qids in entries.items():
    print(name)
    print("base msg")
    show_mean_error(data, qids[0], False)
    print("enhanced msg")
    show_mean_error(data, qids[1], False)
    res = mixed_anova_test(data, 
        withins=qids, valid_col="valid?", within_labels=["base", "enhanced"])
    print("Mixed methods ANOVA test:")
    display(res)
    print()
# %%
# stat tests
""". In addition, these responses spenda significantly longer time (38.7s) in resolving the reCAPTCHA testthan valid responses (8.9s)"""
select = data[data["uid"] == "e0528e46282f4651833a9e22b02a4e6c"]
display(select["Q370_Resolution"].value_counts())
print(select["valid?"].sum())
print(select.shape[0])
print(select["Q198_Page Submit"].mean())
valid = data[data["valid?"] == 1]
print(valid["Q198_Page Submit"].mean())

pg.ttest(select["Q198_Page Submit"], valid["Q198_Page Submit"])

# %%
select["IPAddress"].value_counts().shape
# %%
"""On the other side, compared
with valid responses (196.6s), 
the 44 invalid responses use a significantly shorter time (98.2s) to finish Phase 1.2"""
phase = "Phase 1_2 RT"
print(select[phase].mean())
print(valid[phase].mean())

pg.ttest(select[phase], valid[phase])

phase = "Phase 2 RT"
print(select[phase].mean())
print(valid[phase].mean())

pg.ttest(select[phase], valid[phase])
# %%
display(mwu_valid_invalid_distribution(fdata, "Duration (in seconds)"))


# %%
data[(data["valid?"] == 1) & (data["Attention Check_predict"])][["Q13", "Q49", "Q193"]].fillna(0).sum(axis=1).value_counts()

# %%
data[(data["valid?"] == 1) & (data["Knowledge Check 2_predict"] == True)]["ResponseId"]

# %%
data[(data["valid?"] == 1) & (data["Consistency Check_predict"])]["ResponseId"]

# %%
data["Q193"].dropna().shape

# %%
# inv: 0.18, 0.29, 0.17
# v: 0.5 0.76, 0.4,

# %%
produce_confusion_matrix("Att 1", data, data["Q193"], lambda x: x!=2, )

# %%
xticklabels = ["" for i in range(21)]
xticklabels[0] = "1"
xticklabels[6] = "7"
xticklabels[13] = "14"
xticklabels[20] = "21"
plot_one_question_choices(data, "Q98_1", title="", ylim=[0, 0.26], xticklabels=xticklabels)

# %%
plot_duration(data, "Phase 2 RT", remove_above=1800, title="")

# %%
col = "Phase 1 RT"
print(ttest_valid_invalid_distribution(data, col))
plot_duration(data, col, remove_above=3600, title="")

# %%
col = "Phase 2 RT"
print(ttest_valid_invalid_distribution(data, col))
plot_duration(data, col, remove_above=3600, title="")

# %%

col = "Phase 3 RT"
print(ttest_valid_invalid_distribution(data, col))
plot_duration(data, col, remove_above=3600, title="")

ttest_valid_invalid_distribution(data, "Q3_Page Submit")
ttest_valid_invalid_distribution(data, "Q196_Page Submit")
pprint(res)


plot_duration(data, "Duration (in seconds)", remove_above=7200, title="")

plot_duration(data, "Q363_Page Submit", remove_above=600, title="RT of a page in Phase 2 in Rust Survey")
plot_one_question_choice_place(data, "Q95_1", title="The Choice Dist. of a Likert Scale Question")


# %%
bot = data[data["Resolution_predict"]]
bot


# %% [markdown]
# ## Graphs

# %% [markdown]
# ### System Specifics of Valid vs Invalid

# %%
plot_user_agent(data)
plot_os(data)
plot_screen_resolution(data)

# %%
#plot_max_long_strings(data, prefix)

# %%
#plot_all_pages_durations(data, prefix=prefix)

# %%

#need to find equivalent page for social profile survey
if ANALYZE_RUST:
    plot_duration(data, "Q362_Page Submit", prefix="finished_remove_above_300_", remove_above=300)

# %%
plot_duration(fdata, "Duration (in seconds)")

# %%
plot_duration(fdata, "Duration (in seconds)", remove_above=7200)
# %%
data["Q_SocialSource"].value_counts()

# %%
data[data["valid?"] != 1]["Q_SocialSource"].value_counts()

# %%
data["Duration (in seconds)"].median() / 60

# %%
plot_all_pages_durations(data, prefix, 600)

# %%
plot_all_question_choices(data, prefix)

# %%
mark_all_choice_display_positions(data)

# %%
plot_all_question_choice_places(data, prefix)

# %%
plot_bot_time(data)


# %% [markdown]
# ### Mouse Tracking Analysis

# %%
#Todo 

# %%



