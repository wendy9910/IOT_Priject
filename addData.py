import random
import pandas as pd
import openpyxl
import os

name = random.choice(['Wendy', 'Yang', 'Jan'])

data = pd.read_excel(os.path.join('Staffprofile.xlsx'), engine='openpyxl')
get_name = data['員工姓名'].tolist()
A1 = (data['員工姓名'] == 'Wendy')
A2 = (data["權限"] == "Pass")

A1.to_list()

n = data[(A1 & A2)]

#n = ans
# tolist()
print(n)

list2 = [1, 2, 3, 4, 5]
k = "Pass"

if k in n:
    print('Yes')
#print(data.loc[(data['員工姓名'] == name) & (data['權限'] == 'Pass')])


#file = data[A1 & A2].tolist()


# print(n)
# print(file)

# if name in get_name:
#     print("Yes")


# print(get_name)
