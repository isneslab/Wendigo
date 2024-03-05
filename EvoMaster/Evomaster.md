# Running EvoMaster on DVGA
To conduct our experiments, we made specific modifications to EvoMaster. These modifications included saving each generated query, along with its corresponding response and response time, in a file. Additionally, we adjusted the depth of the generated queries to 5 and 10 for comparison with our agent. Lastly, we excluded the minimization step as it was deemed irrelevant for our experiments. These modifications are provided in patch5.patch and patch10.patch, in which the depth of the generated queries is 5 and 10, respectively.  The following commands can be used to run our setting of EvoMaster on DVGA.

## Setup Evomaster
```
git clone https://github.com/EMResearch/EvoMaster.git
cd EvoMaster
git checkout a617d91af
git patch -p1 < ../patch5.patch # Change to patch10.patch to change the depth to 10
mvn clean install -DskipTests
mv core/target/evomaster.jar .
```

## Run DVGA
```
git clone https://github.com/dolevf/Damn-Vulnerable-GraphQL-Application.git
cd Damn-Vulnerable-GraphQL-Application
docker build -t dvga . 
docker run -d -t -p 5013:5013 -e WEB_HOST=0.0.0.0 --name dvga dvga
```

## Run EvoMaster

`java -jar evomaster.jar --problemType GRAPHQL --bbTargetUrl http://localhost:5013/graphql --blackBox true --outputFormat JAVA_JUNIT_4 --maxTime 6h --ratePerMinute 60`

### Options description

* `--problemType GRAPHQL` - tells EvoMaster to generate GraphQL queries
* `--bbTargetUrl http://localhost:5013/graphql` - tells to EvoMaster the target endpoint
* `--blackBox true` - tells EvoMaster to use black-box mode
* `--outputFormat JAVA_JUNIT_4` - tells EvoMaster to generate JUnit tests (Options: JAVA_JUNIT_5, JAVA_JUNIT_4, KOTLIN_JUNIT_4, KOTLIN_JUNIT_5, JS_JEST, CSHARP_XUNIT)
* `--maxTime 6hs` - tells EvoMaster to run for 6 hours (units: s, m, h -> 1h30m5s is also valid)
* `--ratePerMinute 60` - tells EvoMaster to generate at max 60 queries per minute

## Combine EvoMaster results
```
cd output
python3 combine_EvoMaster_steps.py #Note: fix the num_steps according to the number of output files
```