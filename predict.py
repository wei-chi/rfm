#!/usr/bin/python

"""
example input: day20, format:(0..21, R, F, gt), in 'out_gt'
example output: day20_out, format:(0..21, R, F, gt, R*F), in 'out_RcrossF'
"""

import sys

sum = 0
buckets = [0] * 25

with open(sys.argv[1]) as fp :
	with open(sys.argv[1] + '_out', 'w') as out :
		for line in fp :
			sum += 1
			r = line.split(",")[21]
			f = line.split(",")[22]
			pd = int(r) * int(f)
			buckets[pd - 1] = buckets[pd - 1] + 1
			if pd >= 15 :
				out.write(line[:-1] + ',1\n');
			else :
				out.write(line[:-1] + ',0\n');

print sum
print buckets
