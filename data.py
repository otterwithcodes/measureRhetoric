#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 19:08:06 2022

@author: liuyangdong


用于从 The Variable Persuasiveness of Political Rhetoric 的数据集中生成本项目所使用的数据集。

选取

1. appeal to authority/endorsement
2. appeal to history
3. appeal to national greatness
4. cost/benifit
5. morality
6. public opinion

作为最常见的 Pilitical Rhetoric 的纬度进行测量分类

"""
import csv
import pandas as pd



filename = [
    '1_Building_a_third_runway_at_Heathrow.csv',
    '2_Closing_large_retail_stores_on_Boxing_Day.csv',
    '3_Extending_the_Right_to_Buy.csv',
    '4_Extension_of_surveillance_powers_in_the_UK.csv',
    '5_Fracking_in_the_UK.csv',
    '6_Nationalisation_of_the_railways_in_the_UK.csv',
    '7_Quotas_for_women_on_corporate_boards.csv',
    '8_Reducing_the_legal_restrictions_on_cannabis_use.csv',
    '9_Reducing_university_tuition_fees.csv',
    '10_Renewing_Trident.csv',
    '11_Spending_0.7%_of_GDP_on_overseas_aid.csv',
    '12_Sugar_tax_in_the_UK.csv'
]


p = []

for i in range(len(filename)):
    #print(i)
    #print(filename[i])

    temp = pd.read_csv("./policy_csvs/" + filename[i])
    policy = pd.DataFrame(temp, columns=['policy_id', 'text_one', 'element_one'])
    po = policy.drop_duplicates()
    p.append(po)
    


pl = pd.concat([p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11]])
pl_new = pl.reset_index(drop=True)

dataFile = pl_new.to_csv("dataFile.csv")





