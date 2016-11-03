for (( i=1; i<=20; i++ ))
do
	pop=$(bc <<< "scale=$i; 1/(10^$i)")
	echo $pop
	python main.py 1 -l $pop -e 10000 -s 
done