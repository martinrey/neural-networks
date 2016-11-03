rm resultados.txt
for (( i=1; i<=20; i++ ))
do
	pop=$(bc <<< "scale=3; (1/100)*$i")
	echo $pop
	python main.py 2 -l $pop -e 10000 -s >> resultados.txt
done