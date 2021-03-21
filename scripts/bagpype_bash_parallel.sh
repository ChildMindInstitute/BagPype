#!bin/bashs
batch=$1
path=$(grep 'data_path:' bagpype_template.yaml | tail -n1 | cut -c 13-)
pype=$(grep 'pype_path:' bagpype_template.yaml | tail -n1 | cut -c 13-)
cors=$(grep 'cors:' bagpype_template.yaml | tail -n1 | cut -c7-)
cd "/$pype/BagPype/scripts"
cat "/$path"/${batch}/Output/Results/temp_txtfile.txt | parallel -j "$cors" "python3 /"$pype"/BagPype/scripts/bagpype_clust_function.py {} "${batch}"" 2> ${batch}_log.txt

