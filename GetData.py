import pandas as pd
import openpyxl

data = pd.read_excel("Staffprofile.xlsx")

data.loc[data['部門代號'] == 'A02', '權限'] = "Pass"
data.loc[data['部門代號'] == 'B02', '權限'] = "Error"



print(data)
