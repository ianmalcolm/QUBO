#!/bin/bash
curl -H 'X-DA-Access-Key:NB7wphN6nRhjE94wOhV9j7hWFJKKNI67' \
-H 'Accept: application/json' \
-H 'Content-type:application/json' \
-X POST -d @curl_data.txt \
https://ucp.unicen.smu.edu.sg/da/v2/async/qubo/solve