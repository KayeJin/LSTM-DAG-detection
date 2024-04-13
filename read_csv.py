import csv
import tldextract
# 打开CSV文件
with open('dga_detection_dataset/benign_domain/top-1m.csv', 'r') as csvfile:
    # 创建CSV读取器对象
    csvreader = csv.reader(csvfile)
    y = [x[1] for x in csvreader]
    print(y)
