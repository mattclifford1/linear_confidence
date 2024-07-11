# combine the results into one big table
import os

dir = os.path.dirname(os.path.abspath(__file__))
tables = ['Results-1-Pima Indian Diabetes.txt', 
          'Results-2-Breast Cancer.txt', 
          'Results-3-Hepatitis.txt',
          'Results-4-Heart Disease.txt', 
          'Results-5-MIMIC-III-mortality.txt']
short_names = {'Pima Indian Diabetes': 'Pima Diabetes',
               'Breast Cancer': 'Breast Cancer',
               'Hepatitis': 'Hepatitis',
               'Heart Disease': 'Heart Disease',
               'MIMIC-III-mortality': 'MIMIC ICU'}

# gopen write file
with open(os.path.join(dir, 'combined_table.txt'), 'w') as w_file:
    # get the first table to get table starter
    with open(os.path.join(dir, tables[0]), 'r') as r_table:
        table = r_table.read()
        lines = table.split('\n')
        lines[3] = lines[3][:20] + 'l' + lines[3][20:]
        lines[5] = '& ' + lines[5][:]
        for l in [3, 4, 5]:
            w_file.write(lines[l] + '\n')
    # loop through all tables
    for table in tables:
        with open(os.path.join(dir, table), 'r') as t:
            lines = t.read().split('\n')
            # meta info
            for l in range(0, 3):
                w_file.write(lines[l] + '\n')
            w_file.write('\midrule\n')
            # dataset name
            dataset_name = lines[1].split(' - ')[0][2:]
            w_file.write(
                '\multirow{6}{*}{\\rotatebox{90}{'+short_names[dataset_name]+'}}\n')
            # table info
            for l in range(7, 13):
                w_file.write('& '+ lines[l] + '\n')
    w_file.write('\\bottomrule\n\end{tabular}')
                