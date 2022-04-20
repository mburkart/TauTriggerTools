# TauTagAndProbe
Set of tools to evaluate tau trigger performance on T&amp;P

### How to install

```
cmsrel CMSSW_10_2_20
cd CMSSW_10_2_20/src
cmsenv
git cms-init
git cms-addpkg RecoMET/METFilters
git cms-merge-topic cms-egamma:EgammaPostRecoTools
git clone -o cms-tau-pog git@github.com:cms-tau-pog/TauTriggerTools.git
scram b -j4
```

### How to run

```
cmsRun TauTriggerTools/TauTagAndProbe/test/produceTuples.py inputFiles=DY_2018.root fileNamePrefix=file: period=Run2018 isMC=True runDeepTau=True maxEvents=1000
```

To list all the available options run:
```
python TauTriggerTools/TauTagAndProbe/test/produceTuples.py help
```

### How to submit jobs on CRAB

Submitting task:
```
crab_submit.py --workArea work-area --cfg TauTriggerTools/TauTagAndProbe/test/produceTuples.py --site T2_IT_Pisa --output trigger_tuples TauTagAndProbe/data/2018/DY.txt
```

To list all the available options run:
```
crab_submit.py --help
```

Checking status of the submitted tasks:
```
crab_cmd.py --workArea work-area --cmd status
```
