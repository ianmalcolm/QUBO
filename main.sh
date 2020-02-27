#!/bin/bash
jobid=$(source ./da_script.sh | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")
echo $jobid
status=false
while [ "$status" != "true" ]
do
rawresponse=$(curl -H 'X-DA-Access-Key:NB7wphN6nRhjE94wOhV9j7hWFJKKNI67' -H 'Accept: application/json' -H 'Content-type:application/json' -X GET https://ucp.unicen.smu.edu.sg/da/v2/async/jobs/result/$jobid)
echo $rawresponse
response=$( echo $rawresponse | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")

if [ "$response" = "Done" ] ; then
    status=true
fi
sleep 2
done

curl -H 'X-DA-Access-Key:NB7wphN6nRhjE94wOhV9j7hWFJKKNI67' -H 'Accept: application/json' -H 'Content-type:application/json' -X GET https://ucp.unicen.smu.edu.sg/da/v2/async/jobs/result/$jobid > response.txt