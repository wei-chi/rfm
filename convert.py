#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

#d = {"古董、藝術與礦石" : "1",
#         "圖書與雜誌" : "2",
#         "女包精品與女鞋" : "3",
#         "女裝與服飾配件" : "4",
#         "嬰幼兒與孕婦" : "5",
#         "寵物用品與水族" : "6",
#         "居家、家具與園藝" : "7",
#         "手機與通訊" : "8",
#         "手錶與飾品配件" : "9",
#         "文具與事務用品" : "10",
#         "汽機車精品百貨" : "11",
#         "液晶電視與家電" : "12",
#         "玩具、模型與公仔" : "13",
#         "男性精品與服飾" : "14",
#         "相機、攝影與視訊" : "15",
#         "美容保養與彩妝" : "16",
#         "美食特產與保健" : "17",
#         "運動、戶外與休閒" : "18",
#         "電腦軟硬體與周邊" : "19",
#         "電腦遊戲與主機" : "20",
#         "音響、劇院與MP3" : "21"}

d = {"古董、藝術與礦石"     : "1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
         "圖書與雜誌"       : "0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
         "女包精品與女鞋"   : "0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
         "女裝與服飾配件"   : "0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
         "嬰幼兒與孕婦"     : "0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
         "寵物用品與水族"   : "0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
         "居家、家具與園藝" : "0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
         "手機與通訊"       : "0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0",
         "手錶與飾品配件"   : "0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0",
         "文具與事務用品"   : "0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0",
         "汽機車精品百貨"   : "0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0",
         "液晶電視與家電"   : "0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0",
         "玩具、模型與公仔" : "0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0",
         "男性精品與服飾"   : "0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0",
         "相機、攝影與視訊" : "0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0",
         "美容保養與彩妝"   : "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0",
         "美食特產與保健"   : "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0",
         "運動、戶外與休閒" : "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0",
         "電腦軟硬體與周邊" : "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0",
         "電腦遊戲與主機"   : "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0",
         "音響、劇院與MP3"  : "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"}

sum = 0

with open(sys.argv[1]) as fp :
	with open(sys.argv[1] + '_out', 'w') as out :
		for line in fp :
			sum += 1
			#print line
			pg_catalog_name = line.split(",")[1]
			data = ','.join(line[:-1].split(",")[2:])
			if pg_catalog_name not in d :
				out.write(pg_catalog_name + " " + line)
			else :
				out.write(d[pg_catalog_name] + ',' + data + '\n')

print sum
