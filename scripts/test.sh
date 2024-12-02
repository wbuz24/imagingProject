# Testing script for imaging project written by will buziak

# Runs ../python/fista.py with a varying image size

file=runtime.txt
if [ -e runtime.txt ]
then 
  rm runtime.txt
else 
  echo "$file does not exist"
fi

for i in $( seq 15 100 )
do
  echo "$i"

  python3 python/fista.py $i 
done
