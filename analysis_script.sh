target="/home/brushlab/Dropbox/McLain Leonard/2019_8_Wettability/CA/5 uL droplets"

for file in "$target"/*.avi; do
	f=$(basename "$file")
	d=$(dirname "$file")

	if [ ! -f "$d/results_$f".csv ]; then
		echo "Running on $file"
		python analysis_linear.py "$file" -ss 10
	fi
done