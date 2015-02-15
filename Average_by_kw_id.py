import numpy as np
import csv

def get_conversion_ratio(row):
    try:
        return int(row["APPLICATIONS"])/float(row["CLICKS"])
    except:
        return 0.0

file_name  = "../Data/SEM_DAILY_BUILD.csv"
out_file_name = "kw_data.csv"

kw_dict = {}

# TQS, ITR, CVR

#----Read in Data first time
line_count = 0
click_line_count = 0
kw_counter = 0

input_file = csv.DictReader(open(file_name, "r"))
for row in input_file:

    #Count lines
    line_count += 1

    if(int(row["CLICKS"]) != 0):        
        click_line_count += 1
    else:
        continue            #Skip if no clicks

    if(row["IMPRESSIONS"] == "" or row["IMPRESSIONS"] == "0"):
        continue

    kw_id = int(row["KEYWD_ID"])

    CVR = get_conversion_ratio(row)
    TQS = int(row["TOTAL_QUALITY_SCORE"] if row["TOTAL_QUALITY_SCORE"] != "" else 0)
    ITR = float(row["IMPRESSION_TOTAL_RANK"] if row["IMPRESSION_TOTAL_RANK"] != "" else 0)
    IMP = float(row["IMPRESSIONS"])

    try:
        e = kw_dict[kw_id]
        e[0] += 1
        e[1] += IMP         #Impressions
        e[2] += ITR/IMP     #Average rank
        e[3] += TQS/IMP     #Average quality

        e[4] += ITR         #Total rank
        e[5] += TQS         #Total quality score
        e[6] += CVR         #Conversion ratio
    except:
        kw_dict[kw_id] = [1, IMP, ITR/IMP, TQS/IMP, ITR, TQS, CVR]


with open(out_file_name, 'w') as f:
    f.write("KW_ID, N, AV_IMP, AV_RANK, AV_Q, T_R, T_Q, CVR\n")
    for i in kw_dict:
        f.write("{}, {},".format(i, kw_dict[i][0]))
        for x in range(1,7):
            kw_dict[i][x] /= float(kw_dict[i][0])
            f.write("{},".format(kw_dict[i][x]))
        f.write(",\n")

        

