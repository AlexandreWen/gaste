import os
import sys

import numpy as np
import pandas as pd

# if ".." not in sys.path:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from gaste_test import StratifiedTable2x2

if __name__ == "__main__":
    path = os.path.abspath(os.path.dirname(__file__))
    admission = pd.read_csv(path + "/admission.csv")
    print(admission)
    # Data preparation
    nb_strata = admission.shape[0]
    rows = ("Admitted", "Rejected")
    columns = ("Men", "Women")
    contingency_tables = np.array(
        admission.loc[:, admission.columns != "Department"]
    ).reshape((nb_strata, 2, 2))
    strat_label = ["Department " + dep for dep in admission["Department"]]
    # Create object
    stratified_table = StratifiedTable2x2(
        contingency_tables, labels=strat_label, name_rows=rows, name_columns=columns
    )
    # contingency_tables = [
    #     [[59, 77], [482, 444]],
    #     [[301, 303], [2442, 2405]],
    #     [[3, 4], [190, 196]],
    #     [[22, 10], [128, 67]],
    # ]
    # strat_label = ["ACTT-1 2020", "SOLIDARITY 2020", "Spinner 2020", "Wang 2020"]
    # rows = ("Remdesivir", "Placebo")
    # columns = ("Event", "No event")
    # stratified_table = StratifiedTable2x2(
    #     contingency_tables, labels=strat_label, name_rows=rows, name_columns=columns
    # )
    # Result of analyse
    stratified_table.resume()
    print(
        f"Number of combination in exact calculation of combined p-value : {stratified_table.nb_combination}\n"
    )
    print(
        f"P-value of Cochran-Mantel-Haenszel test : {stratified_table.CMH_test().pvalue}\n"
    )
    print(f"P-value of Breslow-Day test : {stratified_table.BD_test().pvalue}\n")
    print(
        f"P-value of stratified exact test to test under-association : {stratified_table.gaste(alternative='less')}\n"
    )
    print(
        f"P-value of stratified exact test to test under-association (approx gamma) : {stratified_table.gaste(alternative='less', limit_computation_exact=3*10**7)}\n"
    )

    print(
        f"P-value pf stratified exact test to test over-association : {stratified_table.gaste(alternative='greater')}\n"
    )
    print(
        f"One side 'less' Fisher exact test on each stratum : {stratified_table.pval_under}\n"
    )
    print(
        f"One side 'greater' Fisher exact test on each stratum :{stratified_table.pval_over}\n"
    )
    # forest plot
    # stratified_table.plot(thresh_adjust=0.03, save=path + "/meta_analysis_covid.png")
    stratified_table.plot(thresh_adjust=0.03, save=path + "/analysis_admission.png")
