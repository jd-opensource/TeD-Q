python ./example/Many_body_Localization_1D_pennylane.py 20 'cpu' 2>&1 | tee -a result_p_c.txt

python ./example/Many_body_Localization_1D_pennylane.py 20 'gpu2' 2>&1 | tee -a result_p_g.txt 

python ./example/Many_body_Localization_1D_pennylane.py 21 'cpu' 2>&1 | tee -a result_p_c.txt

sleep 60



python ./example/Many_body_Localization_1D_JS.py 21 'cpu' 2>&1 | tee -a result_JS_c.txt 

sleep 60

python ./example/Many_body_Localization_1D_JS.py 14 'cpu' 2>&1 | tee -a result_JS_c.txt 



python ./example/Many_body_Localization_1D_JT.py 21 'gpu1' 2>&1 | tee -a result_JT_g.txt 

python ./example/Many_body_Localization_1D_JT.py 21 'cpu' 2>&1 | tee -a result_JT_c.txt 

sleep 60


python ./example/Many_body_Localization_1D_pennylane.py 21 'gpu2' 2>&1 | tee -a result_p_g.txt 


python ./example/Many_body_Localization_1D_JS.py 21 'gpu1' 2>&1 | tee -a result_JS_g.txt 


python ./example/Many_body_Localization_1D_JT.py 45 'gpu2' 2>&1 | tee -a result_JT_g.txt 

python ./example/Many_body_Localization_1D_JT.py 45 'cpu' 2>&1 | tee -a result_JT_c.txt 

python ./example/Many_body_Localization_1D_JT.py 50 'gpu1' 2>&1 | tee -a result_JT_g.txt 
