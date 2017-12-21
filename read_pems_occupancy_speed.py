import numpy as np
import csv

no = [500011092, 500011102, 500011112, 500011113, 500014111, 500014112]
one_day_occupancy = np.zeros((288, 147))
one_day_speed = np.zeros((288, 147))
csv_reader = csv.reader(open('../Trafficdata/2013pemsd05RawData/d05_text_station_5min_2013_01_01.txt'))
i = 0  # i是one_year_occupancy的行数
j = 0  # j是one_year_occupancy的列数
for row in csv_reader:
    if int(row[1]) in no:
        continue
    try:  # 数据中有缺失暂时是用-1来代替
        one_day_occupancy[i][j] = float(row[10])
    except Exception as e:
        one_day_occupancy[i][j] = -1.0
    try:
        one_day_speed[i][j] = float(row[11])
    except Exception as e:
        one_day_speed[i][j] = -1.0
    j += 1
    if j == 147:
        i += 1
        j = 0
occupancy = np.array(one_day_occupancy)
speed = np.array(one_day_speed)

for m in range(1, 13):
    for d in range(1, 32):
        # 3月10日和9月17日数据不完整，这两天的数据被删除
        if m == 1 and d == 1 or m == 3 and d == 10 or m == 9 and d == 17:
            continue
        if m < 10:
            m_str = '0' + str(m)
        else:
            m_str = str(m)
        if d < 10:
            d_str = '0' + str(d)
        else:
            d_str = str(d)
        try:  # 为了避免程序读到像2月30日这样的月份而停止
            csv_reader = csv.reader(
                open('../Trafficdata/2013pemsd05RawData/d05_text_station_5min_2013_%s_%s.txt' % (m_str, d_str)))
        except Exception as e:
            print('month ' + m_str + ' day ' + d_str + ' does not exist!')
            continue
        i = 0  # i是one_year_occupancy的行数
        j = 0  # j是one_year_occupancy的列数
        for row in csv_reader:
            if int(row[1]) in no:
                continue
            try:
                one_day_occupancy[i][j] = float(row[10])  # 从0开始
            except Exception as e:
                one_day_occupancy[i][j] = -1.0
            try:
                one_day_speed[i][j] = float(row[11])
            except Exception as e:
                one_day_speed[i][j] = -1.0
            j += 1
            if j == 147:
                i += 1
                j = 0
        occupancy = np.vstack((occupancy, one_day_occupancy))
        speed = np.vstack((speed, one_day_speed))
        print('month' + m_str + ' day ' + d_str + ' is finished!')

np.savez('data/pems_speed_occupancy_5min.npz', speed=speed, occupancy=occupancy)
