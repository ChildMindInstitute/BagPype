#!bin/bashs
batch=$1
path=$(grep 'data_path:' "$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/bagpype_template.yaml | tail -n1 | cut -c 13-)
pype=$(grep 'pype_path:' "$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/bagpype_template.yaml | tail -n1 | cut -c 13-)
n_jobs=$(grep 'n_jobs:' "$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/bagpype_template.yaml | tail -n1 | cut -c9-)
cd "/$pype/BagPype/scripts"
cat "/$path"/${batch}/Output/Results/temp_txtfile.txt | parallel -j "$n_jobs" "python3 /"$pype"/BagPype/scripts/bagpype_clust_function.py {} "${batch}"" 2> ${batch}_log.txt
