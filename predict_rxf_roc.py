#!/usr/bin/python

"""
generate ROC curve, compare with gt and R*F value
example input: day20, format:(0..21, R, F, gt)
"""

from __future__ import division
import sys

threshold = 0
TP = 0
FP = 0
TN = 0
FN = 0

data_size = 0
buckets = [0] * 25

while threshold <= 26 :
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	data_size = 0
	with open(sys.argv[1]) as fp :
		with open('threshold_' + str(threshold), 'w') as out :
			for line in fp :
				data_size += 1
				r = line.split(",")[21]
				f = line.split(",")[22]
				gt = line.split(",")[23]
				pd = int(r) * int(f)
				buckets[pd - 1] = buckets[pd - 1] + 1
				if pd >= threshold :
					out.write(line[:-1] + ',1\n');
					if int(gt) == 1 :
						TP += 1
					else :
						FP += 1
				else :
					out.write(line[:-1] + ',0\n');
					if int(gt) == 1 :
						FN += 1
					else :
						TN += 1
#	print 'TP %d, FP %d, FN %d, TN %d' % (TP, FP, FN, TN)
	print 'threshold %d, TPR %.2f, FPR %.2f, accuracy %f' % (threshold, TP/(TP + FN), FP/(FP + TN), (TP + TN)/data_size)
	threshold += 1

print data_size
print buckets

#print 'TP %d, FP %d, FN %d, TN %d' % (TP, FP, FN, TN)
#print 'TPR %.2f, FPR %.2f' % (TP/(TP + FN), FP/(FP + TN))

