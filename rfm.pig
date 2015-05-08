%declare PTIME '2014-11-26';
%declare output_filename 'day25';



A = load '/user/hdfs/ReddoorExportDb002' using PigStorage('\u0001')
as (log_id:long, user_id:chararray, domain_name:chararray,
    url:chararray, ip_address:chararray, dump_time:chararray,
    referer:chararray, session_id:chararray, page_title:chararray,
    sex:chararray, age:chararray, live_city:chararray, ec_id:chararray,
    product_id:chararray, name:chararray, price:chararray,
    ec_catalog_name:chararray, pg_catalog_name:chararray,
    object_name:chararray, brand_name:chararray, web_name:chararray,
    web_type:chararray);

A2 = foreach A generate user_id, pg_catalog_name, dump_time, --GetDay(ToDate(dump_time, 'yyyy-MM-dd HH:mm:ss.SSS')) as day,
                        DaysBetween(ToDate('$PTIME', 'yyyy-MM-dd'), ToDate(SUBSTRING(dump_time, 0, 10), 'yyyy-MM-dd')) as date_diff;
--                      DaysBetween(ToDate('$PTIME', 'yyyy-MM-dd'), ToDate(dump_time, 'yyyy-MM-dd HH:mm:ss.SSS')) as date_diff;
--                      DaysBetween(CurrentTime(), ToDate(dump_time, 'yyyy-MM-dd HH:mm:ss.SSS')) as date_diff;


--A3 = filter A2 by (date_diff >= 0) and (date_diff < 1000000);
A3 = filter A2 by (not pg_catalog_name == 'null')
--   and (date_diff == 1);
--   and (date_diff == 0);
     and (date_diff >= 1) and (date_diff <= 20);
--   and (day <= 20);
--   and user_id matches '02C39E4B-48E4-4A85-B826-F8ED41F4D4D8';
--   and user_id matches '0180A593-BF3B-4069-8F75-5B469E488B5C|102DF51E-60A5-4967-BB29-6654559A40CF';



---------split data
SD = foreach A3 generate $0 .. $3, flatten(STRSPLIT(pg_catalog_name, '\\|'));
SD2 = foreach SD generate $0 .. $3, (bag{tuple()})TOBAG($4 ..) as b2_bag:bag{b2_tuple:tuple()};
SD3 = foreach SD2 generate $0 .. $3, flatten($4);
SD4 = foreach SD3 generate (chararray)$0 as user_id, (chararray)$4 as pg_catalog_name,
     (chararray)$2 as dump_time, (int)$3 as date_diff;

--SD5 = foreach SD4 generate $0 .., 5-($3-1)/4;
--dump SD5;


A4 = group SD4 by (user_id, pg_catalog_name);

A5 = foreach A4 {
    A41 = order SD4 by date_diff asc;
    A42 = limit A41 1;
    generate FLATTEN(A42.user_id) as user_id, FLATTEN(A42.pg_catalog_name) as pg_catalog_name,
             FLATTEN(A42.date_diff) as date_diff, COUNT(SD4) as total;
}

--dump A5;


----------B: find total_max in A5
B = group A5 by user_id;

B2 = foreach B {
    B21 = order A5 by total desc;
    B22 = limit B21 1;
    generate FLATTEN(B22.user_id) as user_id, FLATTEN(B22.total) as total_max;
}


----------join A and B (foreach tuple, add total_max)
C = JOIN A5 by user_id, B2 by user_id;

C2 = foreach C generate A5::user_id, A5::pg_catalog_name, A5::date_diff, A5::total, B2::total_max;

C3 = foreach C2 generate $0 as user_id, $1 as pg_catalog_name,
                         $2 as date_diff, 5-($2-1)/4 as R,
--                       $2 as day, ($2-1)/4+1 as R,
                         $3 as total, $4 as total_max, ($3-1)*5/$4+1 as F;
--C3 = foreach C2 generate $0 as user_id, $1 as pg_catalog_name, 5-($2-1)/4 as R, ($3-1)*5/$4+1 as F;

--dump C3;


/*
-- 1. for day21 and then
C4 = foreach C3 generate user_id, pg_catalog_name, R, F;
store C4 into 'output/$output_filename' using PigStorage(',');
*/





-- 2. for ground truth

----------day21, day22
D = filter A2 by (not pg_catalog_name == 'null')
--    and (day == 21 or day == 22);
    and (date_diff == 0 or date_diff == -1);

--D2 = group D by (user_id, pg_catalog_name);
--D3 = foreach D2 generate group, COUNT(D);
--dump D3;
--D2 = group D by ToDate(SUBSTRING(dump_time, 0, 10), 'yyyy-MM-dd');
--D3 = foreach D2 generate group, COUNT(D);
--dump D3;



---------split data
D2 = foreach D generate $0, flatten(STRSPLIT(pg_catalog_name, '\\|'));
D3 = foreach D2 generate $0, (bag{tuple()})TOBAG($1 ..) as b2_bag:bag{b2_tuple:tuple()};
D4 = foreach D3 generate $0, flatten($1);
D5 = foreach D4 generate (chararray)$0 as user_id, (chararray)$1 as pg_catalog_name; --117548


--S = group D5 by user_id;
--S2 = foreach S generate group, COUNT(D5);
--dump S2;


----------groud_truth
--ground_truth = cogroup C3 by user_id, D5 by user_id;
GT2 = COGROUP C3 by (user_id, pg_catalog_name),D5 by (user_id, pg_catalog_name);
GT3 = filter GT2 by not IsEmpty(C3); --132245

GT4 = foreach GT3 generate flatten(C3.user_id), flatten(C3.pg_catalog_name),
--                           flatten(C3.day),
                           flatten(C3.R),
--                           flatten(C3.total),
--                           flatten(C3.total_max),
                           flatten(C3.F),
                           (IsEmpty(D5)?0:1);

GT5 = order GT4 by $0; --132245
--dump GT5;
store GT5 into 'output/$output_filename' using PigStorage(',');

--GT6 = group GT5 by $0;
--GT7 = foreach GT6 generate group;





-------------------------------------------------
--counting function
/*
Z = group GT5 all;
Z2 = foreach Z generate COUNT(GT5);
dump Z2;

Z3 = group GT7 all;
Z4 = foreach Z3 generate COUNT(GT7);
dump Z4;
*/

--all columns:74153212
--user:60611
--

--not null columns:1784424
--user:34809
--user+cata:195576

--day<=20
--not null columns:1217260
--user:29406
--user+cata:148009 (x)

--day<=20, split catalog
--not null columns:1369524 (SD4)
--user:29046
--user+cata:132245

--0180A593-BF3B-4069-8F75-5B469E488B5C,女裝與服飾配件
