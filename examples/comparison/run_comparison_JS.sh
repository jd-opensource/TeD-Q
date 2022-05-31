python ./example/Many_body_Localization_1D_JS.py 5 'gpu0' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 5 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 6 'gpu1' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 6 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 7 'gpu2' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 7 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 8 'gpu3' 2>&1 | tee -a result_JS_g.txt 

python ./example/Many_body_Localization_1D_JS.py 8 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 9 'gpu0' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 9 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 10 'gpu1' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 10 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 11 'gpu2' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 11 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 12 'gpu3' 2>&1 | tee -a result_JS_g.txt 

python ./example/Many_body_Localization_1D_JS.py 12 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 13 'gpu0' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 13 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 14 'gpu1' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 14 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 15 'gpu2' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 15 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 16 'gpu3' 2>&1 | tee -a result_JS_g.txt 

python ./example/Many_body_Localization_1D_JS.py 16 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 17 'gpu0' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 17 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 18 'gpu1' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 18 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 19 'gpu2' 2>&1 | tee -a result_JS_g.txt &

python ./example/Many_body_Localization_1D_JS.py 19 'cpu' 2>&1 | tee -a result_JS_c.txt &

python ./example/Many_body_Localization_1D_JS.py 20 'gpu3' 2>&1 | tee -a result_JS_g.txt& 

python ./example/Many_body_Localization_1D_JS.py 20 'cpu' 2>&1 | tee -a result_JS_c.txt 

sleep 1200

sh ./example/run_comparison_JT.sh