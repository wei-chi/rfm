[rfm.pig]
1. read raw data from HDFS
2. generate R and F and gt
3. save into HDFS

parameters are declared in files header
only generate one result for each run

ex:
pig rfm.pig


[convert.py]
covert pg_catalog_name into dummy code

input: rfm.pig result

ex:
./convert.py day20


[predict.py]
1. calculate R*F
2. give a threshold, then consider R*F to be 1 or 0

input: convert result
input format: 0..21, R, F, gt
output format: 0..21, R, F, gt, pred(R*F)

ex:
./predict.py day20


[predict_rxf_roc.py]
1. calculate R*F
2. use while loop for each threshold (0.00~0.40)
3. generate 40 point (x:FPR, y:TPR)
4. put the result into EXCEL

input: convert result
output: 40 points in console

ex:
./predict.py day20


[reference.py]
1. calculate logistic regression
2~4. same

IDE: Spyder






