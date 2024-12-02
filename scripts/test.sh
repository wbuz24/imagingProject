# Testing script for imaging project written by will buziak

# Runs ../python/fista.py with a varying image size

file=runtime.txt
if [ -e runtime.txt ]
then 
  rm runtime.txt
  touch runtime.txt
  echo "M: , N: , K: , FISTA runtime (seconds): , MSE: , PSNR: , SSIM:  " >> runtime.txt
else 
  echo "$file does not exist"
fi

for i in $( seq 15 100 )
do
  echo "$i"

  python3 python/testFISTA.py $i 
done
