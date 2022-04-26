# add a new unit, type '# %%'
# add a new markdown unit，type '# %% [markdown]'
# %%
# todo: max string length
# browser version for classification
# 2 sec enforced 


# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.io.stata import value_label_mismatch_doc
import seaborn as sns
import re, json, math
import random
import sqlite3
import pingouin as pg
from pprint import pprint
from collections import Counter
from IPython.display import clear_output, display, Markdown as md
from scipy import stats


# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22
# plt.rcParams["figure.figsize"] = (3,2)
rc = sns.axes_style()
# rc["legend.frameon"] = False
# rc['figure.figsize'] = (3,2)
sns.set(rc=rc)

# %%
#determines what data we runt he analysis on
ANALYZE_RUST = True

#RUST
rust_unfinished_qualtrics_path =  r".\data\[Rust+survey]+www+2022+survey_October+5,+2021_03.20_new.xlsx"
rust_qualtrics_path = rust_unfinished_qualtrics_path
rust_qualtrics_recode_path = r".\data\[Rust+survey]+www+2022+survey_October+10,+2021_21.52_raw_code.xlsx"
rust_trajectory_data_path = r"./data/[Rust survey] www 2022 survey_October 6, 2021_22.59.xml"
rust_expected_referer = "://personal.psu.edu/"
rust_fingerprint_q = "Q370"
rust_query_db_path = r"./data/query-add-missing.db"
rust_phase_questions = {
    "Phase 1_1": [
        "Q8",
        # "Q10",
        "Q14",
        # "Q18",
        # "Q22",
        # "Q24",
        # "Q27",
        # "Q30",
        # "Q32"
        ],
    "Phase 1_2": [
        "Q18",
        "Q22",
        "Q24",
        "Q27",
        "Q30",
        "Q32"
    ],
    "Phase 2": [
        "Q34",
        "Q36",
        "Q48",
        "Q51",
        "Q362",
        "Q39",
        "Q42",
        "Q53",
        "Q55",
        "Q57",
        "Q68",
        "Q363",
        "Q83",
        "Q364",
        "Q99"], 
    "Phase 3": [
        "Q185",
        "Q194"]
}

#DEEPFAKE
deepfake_unfinished_qualtrics_path = "/home/jaron/Projects/bots-ScaleableScams/survey_results/quant/raw_data/study_data/prompt0-4/qualtrics.csv"
deepfake_qualtrics_path =  "/home/jaron/Projects/bots-ScaleableScams/survey_results/quant/processed_data/study_data/prompt0-4/participant_res_filtered_redone2.csv"
deepfake_trajectory_data_path = None #not available yet!
deepfake_expected_referer = "://illinoisdashlab.com"
deekfake_fingerprint_q = "fingerprint_1"
deekfake_query_ip_db_path = "/home/jaron/Projects/bots-ScaleableScams/qualtrics-scripts/data/query-add-missing.db"


#select which info to use for this
unfinished_data_path = rust_unfinished_qualtrics_path if ANALYZE_RUST else deepfake_qualtrics_path
finished_data_path = rust_qualtrics_path if ANALYZE_RUST else deepfake_qualtrics_path
trajectory_data_path = rust_trajectory_data_path if ANALYZE_RUST else deepfake_trajectory_data_path
fingerprint_q = rust_fingerprint_q if ANALYZE_RUST else deekfake_fingerprint_q
query_db_path = rust_query_db_path if ANALYZE_RUST else deekfake_query_ip_db_path
expected_referer = rust_expected_referer if ANALYZE_RUST else deepfake_expected_referer
phase_questions = rust_phase_questions

# %% [markdown]
# ## Functions

# %%
#wrapper, to add results to df
#return (series)

def eval_column_method(eval_df, method, data, column, is_positive, is_missing=lambda x: not x or pd.isna(x), new_suffix="_predict"):
    result_series = produce_confusion_matrix(method, data, data[column], is_positive, is_missing, new_suffix)
    return eval_df.append(result_series, ignore_index=True)

#return data as well with new column with method name
def eval_column_method_two(eval_df, method, data, column, is_positive, is_missing=lambda x: not x or pd.isna(x), new_suffix="_predict"):
    result_series = produce_confusion_matrix(method, data, data[column], is_positive, is_missing, new_suffix)
    data[method] = data[column].apply(lambda x: np.nan if is_missing(x) else is_positive(x))
    return (data, eval_df.append(result_series, ignore_index=True))
    
def calculate_precision(tp, fp, tn, fn):
    return np.float64(tp) / (fp + tp)
def calculate_recall(tp, fp, tn, fn):
    return np.float64(tp) / (tp + fn)
def calculate_f_score(tp, fp, tn, fn):
    return np.float64(tp) / (tp+.5*(fp+fn))

# is_positive: function: elem => true, means "predict as invalid"
def produce_confusion_matrix(method, data: pd.DataFrame, column: pd.Series, is_positive, is_missing=lambda x: not x or pd.isna(x), new_suffix="_predict"):
    missing = column.apply(is_missing)
    invalid_indexes = data["valid?"] != 1
    valid_indexes = data["valid?"] == 1
    valid_missing = missing[valid_indexes].sum()
    invalid_missing = missing[invalid_indexes].sum()
    predictions = column.apply(lambda x: np.nan if is_missing(x) else is_positive(x))
    data[method + new_suffix] = predictions
    fp = predictions[valid_indexes].sum()
    tp = predictions[invalid_indexes].sum()
    invalid_cnt = invalid_indexes.sum() - invalid_missing
    valid_cnt = valid_indexes.sum() - valid_missing
    tn = valid_cnt - fp
    fn = invalid_cnt - tp
    precision = calculate_precision(tp, fp, tn, fn)#np.float64(tp) / (fp + tp)
    recall = calculate_recall(tp, fp, tn, fn)#np.float64(tp) / (tp + fn)
    f_score = calculate_f_score(tp, fp, tn, fn)
    result = pd.Series({"Method": method, "TP": tp, "FP": fp, 
                        "TN": tn, "FN": fn, 
                        "valid missing": valid_missing, "invalid missing": invalid_missing, 
                        "precision": precision, "recall": recall,
                        "f-score" : f_score
                       },
        name=column.name)
    result.to_clipboard(excel=True)
    return result


# %%
parse_qid_data_entry_regex = re.compile(r"(QID\d+): (.+)$", re.MULTILINE)
def parse_qid_data_entry(s: str):
    matches = parse_qid_data_entry_regex.findall(s)
    return dict(matches)


# %%
#functions for marking data

def mark_data(data, column, is_positive, suffix):
    data[f"{column}_{suffix}"] = data[column].apply(is_positive)
    
def mark_element_duplication(data, column):
    counts = data[column].value_counts()

    def has_duplication(x):
        return counts[x] > 1 if type(x) == str else True
    
    mark_data(data, column, has_duplication, "duplicate")  
    
def qid_data_mark_duplicate(data: pd.DataFrame, column):
    rows = data[column].apply(lambda x: "" if type(x) != str else x).apply(parse_qid_data_entry).apply(lambda x: set(x.values()))
    counter = Counter()
    for row in rows:
        counter.update(row)
    
    def find_multi_duplicate(row):
        if len(row) == 0:
            # no element. abnormacy. return problematic
            return True
        else:
            for elem in row:
                if counter.get(elem) > 1:
                    # find duplicant
                    return True
        # normal
        return False
    rows = rows.apply(find_multi_duplicate)
    data[f"{column}_duplicate"] = rows

# %%
def get_major_version(s: str):
    if type(s) == str and s:
        return int(s.split(".")[0])
    return 0


def plot_user_agent(data):
    bins = [i for i in range(0, 101,10)]
    browser_col = f'{fingerprint_q}_Browser'
    version_col = f"{fingerprint_q}_Version"
    versions = pd.cut(data[data["valid?"] == 1][version_col].apply(get_major_version), bins)
    new_data = pd.DataFrame({
        "browser": data[browser_col], 
        "version": versions
    })
    valid_counts = new_data.value_counts(normalize=True)
    versions = pd.cut(data[data["valid?"] != 1][version_col].apply(get_major_version), bins)
    new_data = pd.DataFrame({
        "browser": data[browser_col], 
        "version": versions
    })
    invalid_counts = new_data.value_counts(normalize=True)
    to_plot = pd.DataFrame({
        "valid": valid_counts, 
        "invalid": invalid_counts
    })
    to_plot.sort_index(inplace=True)
    to_plot.plot(kind="bar")

def count_browser_versions(data: pd.DataFrame):
    bins = [i for i in range(0, 101,10)]
    browser_col = f'{fingerprint_q}_Browser'
    version_col = f"{fingerprint_q}_Version"
    versions = pd.cut(data[version_col].apply(get_major_version), bins)
    new_data = pd.DataFrame({
        "browser": data[browser_col], 
        "version": versions
    })
    counts = new_data.value_counts(normalize=False)
    counts.sort_index(inplace=True)
    return counts

def plot_os(data):
    valid_data = data[data["valid?"] == 1]
    invalid_data = data[data["valid?"] != 1]
    to_plot = pd.DataFrame({
        "valid": valid_data[f"{fingerprint_q}_Operating System"].value_counts(normalize=True), 
        "invalid": invalid_data[f"{fingerprint_q}_Operating System"].value_counts(normalize=True)
    })
    to_plot.sort_index(inplace=True)
    to_plot.plot(kind="bar")

def plot_screen_resolution(data):
    valid_data = data[data["valid?"] == 1]
    invalid_data = data[data["valid?"] != 1]
    to_plot = pd.DataFrame({
        "valid": valid_data[f"{fingerprint_q}_Resolution"].value_counts(normalize=True), 
        "invalid": invalid_data[f"{fingerprint_q}_Resolution"].value_counts(normalize=True)
    })
    to_plot.sort_index(inplace=True)
    to_plot.plot(kind="bar")

def get_ip_timezone(ip: str, query_db_path):
    with sqlite3.connect(query_db_path) as conn:
        cur = conn.cursor()
        cur.execute('''
        SELECT MinFraud FROM Queries
        WHERE IP=?''', 
        (ip,))
        minfraud = cur.fetchone()
        if not minfraud:
            return None
        minfraud = json.loads(minfraud[0])
        return minfraud["response"]["ip_address"]["location"]["time_zone"]
    
def server_has_uid(uid: str):
    pass

def get_queries_results(ip:str, query_db_path):
    with sqlite3.connect(query_db_path) as conn:
        cur = conn.cursor()
        cur.execute('''
        SELECT VirusTotal, MinFraud, IPRegistry, DNSBL FROM Queries
        WHERE IP=?''', 
        (ip,))
        row = cur.fetchone()
        if not row:
            return None
        result = {}
        # int
        result["VirusTotal"] = json.loads(row[0])["response"]["attributes"]["last_analysis_stats"]["malicious"]
        # float
        result["MinFraud"] = json.loads(row[1])["response"]["ip_address"]["risk"]
        # bool
        result["IPRegistry"] = json.loads(row[2])["response"]["security"]["is_threat"]
        result["DNSBL"] = json.loads(row[3])["response"]["blacklisted"]
        return result

def predict_with_ip_databases(data, query_db_path):
    results = data["IPAddress"].apply(get_queries_results, query_db_path=query_db_path)
    data["VirusTotal_predict"] = results.apply(lambda x: np.nan if pd.isna(x) else x["VirusTotal"] > 0)
    data["MinFraud_predict"] = results.apply(lambda x: np.nan if pd.isna(x) else x["MinFraud"] > 5.0)
    data["IPRegistry_predict"] = results.apply(lambda x: np.nan if pd.isna(x) else x["IPRegistry"])
    data["DNSBL_predict"] = results.apply(lambda x: np.nan if pd.isna(x) else x["DNSBL"])


def plot_duration(data: pd.DataFrame, column: str, prefix="", remove_above=None, title=None):
    plt.clf()
    if remove_above:
        data = data[data[column] <= remove_above]
    valid = data[data["valid?"] == 1][column]
    invalid = data[data["valid?"] != 1][column]
    out = pd.DataFrame({"valid": valid, "invalid": invalid})
    ax = sns.histplot(out, stat="probability", bins=40, element="step", common_norm=False)
    ax.set_xlabel("Duration (second)")
    if title == None:
        title = f"Duration Distribution of {prefix}{column}"
    if title != "":
        ax.set_title(title)
    # plt.tight_layout()
    # ax.legend(frameon=False)
    plt.savefig(f"./figures/{prefix}{column}.pdf", bbox_inches="tight")


def plot_all_pages_durations(data, prefix="", remove_above=None):
    durations = data.filter(axis="columns", like="Page Submit")
    if remove_above:
        prefix += f"remove_above_{remove_above}_"
    for column in durations.columns:
        plot_duration(data, column, prefix, remove_above)

def plot_one_question_choices(data: pd.DataFrame, column: str, prefix="", title=None, ylim=None, xticklabels=None):
    plt.clf()
    valid = data[data["valid?"] == 1][column].dropna().astype('int32')
    invalid = data[data["valid?"] != 1][column].dropna().astype('int32')
    valid_counts = valid.value_counts(normalize=True, sort=False)
    invalid_counts = invalid.value_counts(normalize=True, sort=False)
    to_plot = pd.DataFrame({"valid": valid_counts, "invalid": invalid_counts})
    to_plot.sort_index(inplace=True)
    ax = to_plot.plot(kind="bar")
    if title == None:
        title = f"{prefix}{column} Choice Distribution"
    if title != "":
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Choice")
    ax.tick_params(axis="x", labelrotation=0)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./figures/{prefix}{column}_choices.pdf", bbox_inches="tight")
    plt.close(ax.figure)

#data: with choice code
def plot_all_question_choices(data: pd.DataFrame, prefix=""):
    cols = find_choice_question_columns(data)
    for col in cols:
        plot_one_question_choices(data, col, prefix)
        
def find_choice_display_position(row, column):
    choice = row[column]
    if pd.isna(choice):
        return np.nan
    if f"{column}_DO" in row.index:
        choices = row[f"{column}_DO"].split("|")
        return choices.index(str(round(choice))) + 1
    else:
        return choice

def mark_choice_display_position(data, column):
    dp_col = f"{column}_display_position"
    data[dp_col] = data.apply(find_choice_display_position, axis=1, column=column)

def find_max_long_string(row: pd.Series, indexes):
    indexes = iter(indexes)
    cur_position = next(indexes)
    cur_len = 1
    max_position = cur_position
    max_len = cur_len
    for index in indexes:
        position = row[index]
        if pd.isna(position):
            continue
        if position == cur_position:
            cur_len += 1
        else:
            cur_position = position
            cur_len = 1
        if max_position != cur_position:
            if max_len < cur_len:
                max_position = cur_position
                max_len = cur_len
        else:
            max_len = cur_len
    return max_len, max_position
 

def find_all_max_long_strings(data: pd.DataFrame):
    indexes = list(filter(lambda x: x.endswith('_display_position'), data.columns))
    results = data.apply(find_max_long_string, axis=1, indexes=indexes)
    data["max_string_length"] = results.apply(lambda x: x[0])
    data["max_string_choice_position"] = results.apply(lambda x: x[1])

def find_nasa_tlx_max_long_strings(data: pd.DataFrame):
    tlx1 = [f"Q{i}_1_display_position" for i in range(77, 83)]
    tlx2 = [f"Q{i}_1_display_position" for i in range(93, 99)]

    results = data.apply(find_max_long_string, axis=1, indexes=tlx1)
    data["TLX1_max_string_length"] = results.apply(lambda x: x[0])
    data["TLX1_max_string_choice_position"] = results.apply(lambda x: x[1])
    results = data.apply(find_max_long_string, axis=1, indexes=tlx2)
    data["TLX2_max_string_length"] = results.apply(lambda x: x[0])
    data["TLX2_max_string_choice_position"] = results.apply(lambda x: x[1])

def plot_max_long_string(data: pd.DataFrame, column: str, prefix: str):
    plt.clf()
    valid = data[data["valid?"] == 1][column].dropna().astype('int32')
    invalid = data[data["valid?"] != 1][column].dropna().astype('int32')
    valid_counts = valid.value_counts(normalize=True, sort=False)
    invalid_counts = invalid.value_counts(normalize=True, sort=False)
    to_plot = pd.DataFrame({"valid": valid_counts, "invalid": invalid_counts})
    to_plot.sort_index(inplace=True)
    ax = to_plot.plot(kind="bar")
    ax.set_title(f"{prefix} {column}")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Max String Length")
    plt.tight_layout()
    plt.savefig(f"./figures/{prefix}{column}.pdf")
    plt.close(ax.figure)

def plot_max_long_strings(data, prefix: str):
    if "max_string_length" not in data.columns:
        mark_all_choice_display_positions(data)
    find_all_max_long_strings(data)
    find_nasa_tlx_max_long_strings(data)
    plot_max_long_string(data, "max_string_length", prefix)
    plot_max_long_string(data, "TLX1_max_string_length", prefix)
    plot_max_long_string(data, "TLX2_max_string_length", prefix)

def plot_one_question_choice_place(data, column, prefix="", title=None):
    dp_col = f"{column}_display_position"
    if dp_col not in data.columns:
        mark_choice_display_position(data, column)
    plt.clf()
    valid = data[data["valid?"] == 1][dp_col].dropna().astype('int32')
    invalid = data[data["valid?"] != 1][dp_col].dropna().astype('int32')
    valid_counts = valid.value_counts(normalize=True, sort=False)
    invalid_counts = invalid.value_counts(normalize=True, sort=False)
    to_plot = pd.DataFrame({"valid": valid_counts, "invalid": invalid_counts})
    to_plot.sort_index(inplace=True)
    ax = to_plot.plot(kind="bar")
    if title == None:
        title = f"{prefix}{column} Choice Place Distribution"
    if title != "":
        ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Choice Place")
    plt.tight_layout()
    plt.savefig(f"./figures/{prefix}{column}_choice_places.pdf")
    plt.close(ax.figure)

def mark_all_choice_display_positions(data):
    cols = find_choice_question_columns(data)
    for col in cols:
        mark_choice_display_position(data, col)

#data: with choice code
def plot_all_question_choice_places(data: pd.DataFrame, prefix=""):
    cols = find_choice_question_columns(data)
    for col in cols:
        plot_one_question_choice_place(data, col, prefix)

def parse_resolution(s: str):
    if not s:
        return None
    result = s.split("x")
    return [int(s) for s in result]

# invalid if it has odd resolution
def is_odd_resolution(res: list):
    if not res:
        return np.nan
    for i in res:
        if i % 2:
            return True
    # horizon < vertical
    # if res[0] < res[1]:
    #     return True
    return False


def evaluate_odd_resolution(data):
    method = "Resolution"
    series = data[f"{fingerprint_q}_Resolution"].apply(parse_resolution).apply(is_odd_resolution)
    return produce_confusion_matrix(method, data, series, lambda x: x, is_missing=pd.isna)

# %%
def parse_qid_json(data: dict):

    def find_end(s):
        i = len(s) - 1
        while i >= 0:
            if s[i] == "}":
                return i
            i -= 1
        return -1

    new_data = {}
    for qid, s in data.items():
        try:
            obj = json.loads(s)
        # deal with imcomplete data
        except Exception:
            end = find_end(s)
            if end >= 0:
                s = s[:end+1] + "]"
                obj = json.loads(s)
        new_data[qid] = obj
    return new_data

# %%
#trajectory functions

def find_trajectory_max_speed(positions): # unit: pixel/ms
    if not positions:
        return math.nan
    max_speed = 0
    for i in range(len(positions) - 1):
        interval = positions[i+1]["time"] - positions[i]["time"]
        if interval == 0:
            speed = 0.0
        else:
            distance = math.sqrt((positions[i+1]["x"] - positions[i]["x"]) ** 2 + 
                (positions[i+1]["y"] - positions[i]["y"]) ** 2)
            speed = distance / interval
        if speed > max_speed:
            max_speed = speed
    return max_speed

def find_trajectory_sequence_length(positions):
    if not positions:
        return math.nan
    def is_same_position(pos1, pos2):
        return pos1["x"] == pos2["x"] and pos1["y"] == pos2["y"]
        
    if len(positions) == 0:
        return 0
    length = 1
    for i in range(1, len(positions)):
        if not is_same_position(positions[i], positions[i-1]):
            length += 1
    return length

def find_trajectory_path(positions):
    if not positions:
        return math.nan
    journey = 0.0
    for i in range(len(positions) - 1):
        distance = math.sqrt((positions[i+1]["x"] - positions[i]["x"]) ** 2 + 
            (positions[i+1]["y"] - positions[i]["y"]) ** 2)
        journey += distance
    return journey

# %%
def qid_data_different_item_number_within_response(data, column):
    series = (data[column]
        .apply(lambda x: "" if type(x) != str else x)
        .apply(parse_qid_data_entry)
        .apply(lambda x: set(x.values()))
        .apply(lambda x: len(x)))
    return series.value_counts()

def qid_data_show_different_within_response(data, column):
    return data[data[column]
        .apply(lambda x: "" if type(x) != str else x)
        .apply(parse_qid_data_entry)
        .apply(lambda x: len(set(x.values()))) > 1]

# %%
#Functions for confusion matricies

def element_is_blank(data: pd.DataFrame, method, column: str):
    series = data[column].copy()
    series.name = f"{column}_blank"
    return produce_confusion_matrix(method, data, series, is_positive=pd.isna, is_missing=lambda x: False)

# not include NA in positive/negative
def elements_has_duplication(data: pd.DataFrame, method, column: str):
    counts = data[column].value_counts()
    #method = f'{column} Duplicate'
    def has_duplication(x):
        return counts[x] > 1 if type(x) == str else True
    
    series = data[column].copy()
    series.name = f"{column}_duplicate"
    return produce_confusion_matrix(method, data, series, has_duplication, is_missing=pd.isna)

def predict_on_trajectory_path(data: pd.DataFrame, col: str):
    method = f"Trajectory path length outlier"
    paths = data[col]
    predict = stats.zscore(paths).abs() > 2
    return produce_confusion_matrix(method, data, predict, lambda x: x, pd.isna)

def qid_data_find_duplicate(data: pd.DataFrame, method, column):
    #method = f'{column} Duplicate'
    rows = data[column].apply(lambda x: "" if type(x) != str else x).apply(parse_qid_data_entry).apply(lambda x: set(x.values()))
    missing = rows.apply(lambda x: len(x) == 0 or 'true' in x)
    counter = Counter()
    for row in rows:
        counter.update(row)
    
    def find_multi_duplicate(row):
        if len(row) == 0:
            # no element. abnormacy. return problematic
            return True
        else:
            for elem in row:
                if counter.get(elem) > 1:
                    # find duplicant
                    return True
        # normal
        return False

    rows = rows.apply(find_multi_duplicate)
    rows[missing] = np.nan
    result = produce_confusion_matrix(method, data, rows, lambda x: x, is_missing=pd.isna)
    return result

def find_choice_question_columns(data: pd.DataFrame):
    data = data.filter(axis="columns", regex=r"^Q\d+(_1)?$")
    cols = []
    for col in data.columns:
        if data[col].dtype == np.float64:
            cols.append(col)
    return cols

def show_os_timezone_comparison(data: pd.DataFrame):
    (data[data["Timezone mismatch_predict"] == True][["IPTimeZone", "processed_clientOSTimezone"]]
        .value_counts()
        .to_clipboard(excel=True))

def putNullforNaNs(x):
    if pd.isna(x):
        return "null"
    return x

def parse_first_val(x):
    if len(x) == 0:
        return np.nan
    else:
        return list(x.values())[0]

def compare_webrtc_ip(data):
    method = "WebRTC-different"
    series = data["webrtcIP"].apply(putNullforNaNs).apply(parse_qid_data_entry).apply(lambda x: set() if not x else set(list(x.values())[0].split(", ")))
    missing = series.apply(lambda x: len(x) == 0)
    new_series = []
    for row in series.index:
        if data["IPAddress"].loc[row] in series.loc[row]:
            new_series.append(False)
        else:
            new_series.append(True)
    new_series = pd.Series(new_series, index=series.index, name="webrtcIP")
    new_series[missing] = np.nan
    result = produce_confusion_matrix( method, data, new_series, lambda x: x, is_missing=pd.isna)
    return result

def compare_os_timezone_with_ip(data: pd.DataFrame, query_db_path: str):
    method = "Timezone mismatch"
    column = "clientOSTimezone"
    ips = data["IPAddress"]
    ip_tzs = []
    for ip in ips:
        tz = get_ip_timezone(ip, query_db_path)
        ip_tzs.append(tz)
    ip_tzs = pd.Series(ip_tzs, index=ips.index)
    data["IPTimeZone"] = ip_tzs
    missing = ip_tzs.isna()
    ostimezones = data[column].apply(parse_qid_data_entry).apply(parse_first_val)
    data["processed_clientOSTimezone"] = ostimezones
    result = ip_tzs.ne(ostimezones)
    result[missing] = np.nan
    result.name = column
    return produce_confusion_matrix(method, data, result, is_positive=lambda x: x, is_missing=pd.isna)
    
def plot_bot_time(data: pd.DataFrame, remove_above=7200):
    plt.clf()
    column = "Duration (in seconds)"
    if remove_above:
        data = data[data[column] <= remove_above]
    bot_indexes = data[f"{fingerprint_q}_Resolution"].apply(parse_resolution).apply(is_odd_resolution)
    invalid_indexes = data["valid?"] != 1
    nonbot_invalid_indexes = invalid_indexes ^ bot_indexes & invalid_indexes
    nonbot_invalid = data[nonbot_invalid_indexes][column]
    valid = data[data["valid?"] == 1][column]
    invalid = data[invalid_indexes][column]
    bot_all = data[bot_indexes]
    valid_and_bot = bot_all.append(data[data["valid?"] == 1])
    bot = bot_all[column]
    print(pg.ttest(nonbot_invalid, bot, ))
    out = pd.DataFrame({
        "valid": valid, 
        "nonbot invalid": nonbot_invalid, 
    "bot": bot})
    ax = sns.histplot(out, stat="probability", bins=40, element="step", common_norm=False)
    ax.set_xlabel("Duration (second)")
    print("RT < 1200, bot within invalid prop:", (bot < 1200).sum() / (invalid < 1200).sum())
    print("RT < 1200, bot prop:",  (bot < 1200).sum() / len(bot))
    title = f"Duration Distribution of {prefix}{column}"
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"./figures/{prefix}{column}_bot.png")
    plot_user_agent(valid_and_bot)

def compare_webrtc_ip_missing_as_invalid(data):
    method = "WebRTC"
    series = data["webrtcIP"].apply(putNullforNaNs).apply(parse_qid_data_entry).apply(lambda x: set() if not x else set(list(x.values())[0].split(", ")))
    # missing = series.apply(lambda x: len(x) == 0)
    new_series = []
    for row in series.index:
        if data["IPAddress"].loc[row] in series.loc[row]:
            new_series.append(False)
        else:
            new_series.append(True)
    new_series = pd.Series(new_series, index=series.index, name="webrtcIP_different_or_missing")
    # new_series[missing] = np.nan
    result = produce_confusion_matrix( method, data, new_series, lambda x: x, is_missing=lambda x: False)
    return result

# old version, compare previous cookie existance
# def compare_cookie(data):
#     method = 'Cookie'
#     series = data["cookie"].apply(putNullforNaNs).apply(parse_qid_data_entry).apply(lambda x: "" if not x else list(x.values())[0])
#     missing = series.apply(lambda x: x == "")
#     new_series = series.ne(data['ResponseId'])
#     new_series[missing] = np.nan
#     new_series.name = "cookie"
#     result = produce_confusion_matrix(method, data, new_series, lambda x: x, is_missing=pd.isna)
#     return result

# new version, compare content duplication
def compare_cookie(data):
    return qid_data_find_duplicate(data, "Cookie", "cookie")

def mouse_moved_evaluation(data, column):
    series = (data[column]
        .apply(lambda x: "" if type(x) != str else x)
        .apply(parse_qid_data_entry)
        .apply(lambda x: set(x.values())))
    missing = series.apply(lambda x: len(x) == 0)
    series[missing] = np.nan
    result = produce_confusion_matrix('Mouse Moved', data, series, lambda x: len(x) == 0 or 'false' in x)
    return result    

def merge_raw_data(labeled_data: pd.DataFrame, raw_data: pd.DataFrame):
    labeled_data = labeled_data[["ResponseId", "valid?"]]
    return labeled_data.merge(raw_data, on="ResponseId")


# %%
def count_qids(series: pd.Series):
    counter = Counter()
    for response in series:
        counter.update(response.keys())
    return counter

def compute_response_RT_skew_kurtosis(data: pd.DataFrame, is_bot=lambda x: x[f"Resolution_predict"]):
    valid = data["valid?"] == 1
    invalid = data["valid?"] != 1
    bot = data.apply(is_bot, axis=1) & invalid
    nonbot_invalid = invalid ^ bot
    data = data.filter(axis="columns", like="_Page Submit")
    valid_o = data[valid]
    nb_invalid_o = data[nonbot_invalid]
    bot_o = data[bot]
    # if filter_outlier:
    #     valid = valid_o[stats.zscore(valid_o).abs() <= 3]
    #     nb_invalid = nb_invalid_o[stats.zscore(nb_invalid_o).abs() <= 3]
    #     bot = bot_o[stats.zscore(bot_o).abs() <= 3]
    #     print(f"filtered out {len(valid_o) - len(valid)} valid responses")
    #     print(f"filtered out {len(nb_invalid_o) - len(nb_invalid)} nonbot invalid responses")
    #     print(f"filtered out {len(bot_o) - len(bot)} valid responses")
    # else:
    #     valid = valid_o 
    #     nb_invalid = nb_invalid_o
    #     bot = bot_o
    valid = valid_o 
    nb_invalid = nb_invalid_o
    bot = bot_o
    valid_skews = valid.apply(lambda row: row.skew(), axis=1)
    valid_kurts = valid.apply(lambda row: row.kurtosis(), axis=1)
    nb_invalid_skews = nb_invalid.apply(lambda row: row.skew(), axis=1)
    nb_invalid_kurts = nb_invalid.apply(lambda row: row.kurtosis(), axis=1)
    bot_skews = bot.apply(lambda row: row.skew(), axis=1)
    bot_kurts = bot.apply(lambda row: row.kurtosis(), axis=1)
    valid_res = pd.DataFrame({"skew": valid_skews, "kurtosis": valid_kurts})
    nb_invalid_res = pd.DataFrame({"skew": nb_invalid_skews, "kurtosis": nb_invalid_kurts})
    bot_res = pd.DataFrame({"skew": bot_skews, "kurtosis": bot_kurts})
    return valid_res, nb_invalid_res, bot_res
# %%
#functions for new columns

#check whether the two questions for consistency match
def consistent_answers(row):
    return row['consistency_1'] == row['consistency_2']

#check whether the two questions for consistency match
def rust_consistent_answers(row):
    if pd.isna(row["Q20"]) or pd.isna(row["Q191"]):
        return np.nan
    return row['Q20'] == row['Q191']

#check whether any of the attention checks were right
def attention_correct(row, right_answer=5):
    if row['att_1'] == right_answer or row['att_2'] == right_answer or row['att_3'] == right_answer:
        return True
    else:
        return False

#check whether any of the attention checks were right
def rust_attention_correct(row, right_answer=2): 
    if pd.isna(row['Q13']) and pd.isna(row['Q49']) and pd.isna(row['Q193']):
        return np.nan
    if row['Q13'] == right_answer or row['Q49'] == right_answer or row['Q193'] == right_answer:
        return True
    else:
        return False

def rust_knowledge1_correct(row): 
    if pd.isna(row['Q44']) or pd.isna(row['Q47']):
        return np.nan
    return (row['Q44'] == 2) and (row['Q47'] == 3)

def rust_knowledge2_correct(row):
    if pd.isna(row['Q38']) and pd.isna(row['Q41']):
        return np.nan
    return (row['Q38'] == 1) or (row['Q41'] == 4)

# return: True if rejected (i.e., True if distribution of valid/invalid is different)
def ttest_valid_invalid_distribution(data: pd.DataFrame, column: str, p_thres=0.05):
    valid = data[data["valid?"] == 1][column]
    invalid = data[data["valid?"] != 1][column]
    res = pg.ttest(valid, invalid)
    return res["p-val"].iloc[0] <= p_thres, res

# return: True if rejected (i.e., True if distribution of valid/invalid is different)
def mwu_valid_invalid_distribution(data: pd.DataFrame, column: str, p_thres=0.05):
    valid = data[data["valid?"] == 1][column]
    invalid = data[data["valid?"] != 1][column]
    res = pg.mwu(valid, invalid)
    return res["p-val"].iloc[0] <= p_thres, res

def ttest_all_timings(data: pd.DataFrame):
    import itertools
    columns = list(itertools.chain(*phase_questions.values()))
    res = {}
    for col in columns:
        col = f"{col}_Page Submit"
        res[col] = ttest_valid_invalid_distribution(data, col)
    return res

def group_ttest_RT_in_phases(data: pd.DataFrame):
    res = ttest_all_timings(data)
    phases = {}
    for phase, questions in phase_questions.items():
        new_res = []
        for q in questions:
            new_res.append(res[f"{q}_Page Submit"])
        phases[phase] = new_res
    return phases, res

def calculate_phase_RT(data: pd.DataFrame, phase_questions):

    def calculate_row(row: pd.Series, questions):
        s = 0.0
        for q in questions:
            t = row[f"{q}_Page Submit"]
            if not pd.isna(t):
                s += t
        return s

    for phase, questions in phase_questions.items():
        rt = data.apply(calculate_row, axis=1, questions=questions)
        data[f"{phase} RT"] = rt
        
def kleene_or_columns(data: pd.DataFrame, cols: list):
    cols = iter(cols)
    result = data[next(cols)].convert_dtypes()
    for col in cols:
        result |= data[col].convert_dtypes()
    return result

# only one within factor is accepted
def mixed_anova_test(data: pd.DataFrame, withins: list, valid_col: str, within_labels: list):
    new_data = []
    for _, row in data.iterrows():
        for i, within in enumerate(withins):
            new_item = {"subject": row["ResponseId"]}
            new_item["dv"] = row[within]
            new_item["within"] = within_labels[i]
            new_item["between"] = "valid" if row[valid_col] else "invalid"
            new_data.append(new_item)
    new_data = pd.DataFrame(new_data)
    # new_data["subject"] = new_data["subject"].astype("category")
    # new_data["within"] = new_data["within"].astype("category")
    # new_data["between"] = new_data["between"].astype("category")
    return pg.mixed_anova(new_data, dv="dv", within="within", between="between", subject="subject")



def show_mean_error(data: pd.DataFrame, col: str, filter_outlier=True):
    if filter_outlier:
        data_new = data[stats.zscore(data[col]).abs() <= 3]
        print(f"filter out {len(data) - len(data_new)} responses with zscore")
        data = data_new
    invalid = data[data["valid?"] != 1][col]
    valid = data[data["valid?"] == 1][col]
    # df = pd.DataFrame(columns = ["mean", "std error"])
    # df = df.append(pd.Series({"mean": valid.mean(), "std error": valid.sem()}, name="Valid"))
    # df = df.append(pd.Series({"mean": invalid.mean(), "std error": invalid.sem()}, name="Invalid"))
    df = pd.DataFrame(columns = ["mean (std error)"])
    df = df.append(pd.Series({"mean (std error)": f"{valid.mean():.3f} ({valid.sem():.3f})"}, name="Valid"))
    df = df.append(pd.Series({"mean (std error)": f"{invalid.mean():.3f} ({invalid.sem():.3f})"}, name="Invalid"))
    df.name = col
    df.to_clipboard(excel=True)
    print(col)
    display(df)
    print("T test:")
    display(ttest_valid_invalid_distribution(data, col)[1])
    print()
    return df, data

# %% [markdown]
# ## Load Data and Process Data
