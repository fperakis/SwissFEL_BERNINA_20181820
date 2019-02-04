
start=$1
end=$2

if [ $# -eq 0 ];
then
  echo "give a single argument: run number to start at"
else

  echo "${start} --> ${end}"
  for i in `seq ${start} ${end}`; do
      echo $i
      salloc -n 12 mpirun ./process.py -r ${i} -s 17998 -t 0.04
  done
fi

