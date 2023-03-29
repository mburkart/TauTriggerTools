import re

def readFileList(fileList, inputFileName, fileNamePrefix):
    """read intput file list from a text file"""
    inputFile = open(inputFileName, 'r')
    for name in inputFile:
        if len(name) > 0 and name[0] != '#':
            fileList.append(fileNamePrefix + name)
    inputFile.close()

def addFilesToList(fileList, inputFiles, fileNamePrefix):
    """read intput file list from a another list"""
    for name in inputFiles:
        if len(name) > 0 and name[0] != '#':
            fileList.append(fileNamePrefix + name)

def getYear(period):
    """returns year for the given datataking period"""
    match = re.search(r'^Run([0-9]+)', period)
    if match is None:
        raise RuntimeError('Unable to extract year from the datataking period = "{}"'.format(period))
    return int(match.group(1))

def getGlobalTag(period, isMC):
    """ Returns global tag that should be used to run tuple production with 102X release.
        The recommended global tag values are taken from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
    """
    globalTags_data = {
        'Run2016': '102X_dataRun2_nanoAOD_2016_v1',
        'Run2017': '102X_dataRun2_v8',
        'Run2018ABC': '102X_dataRun2_Sep2018Rereco_v1',
        'Run2018D': '102X_dataRun2_Prompt_v12'
    }

    globalTags_mc = {
        'Run2016': '102X_mcRun2_asymptotic_v6',
        'Run2017': '102X_mc2017_realistic_v6',
        'Run2018': '102X_upgrade2018_realistic_v18',
    }

    tags = globalTags_mc if isMC else globalTags_data

    if period not in tags:
        raise RuntimeError("Global tag for period '{}' in not found.".format(period))
    return tags[period]

def getBtagThreshold(period, wp):
    """ Returns DeepFlavour btag threshold for a given period and working point.
        The values are taken from https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation
    """
    btag_thresholds = {
        'Run2016': { 'Loose': 0.0614, 'Medium': 0.3093, 'Tight': 0.7221 },
        'Run2017': { 'Loose': 0.0521, 'Medium': 0.3033, 'Tight': 0.7489 },
    }
    btag_thresholds.update(dict.fromkeys(['Run2018', 'Run2018ABC', 'Run2018D'],
        { 'Loose': 0.0494, 'Medium': 0.2770, 'Tight': 0.7264 }))

    if period not in btag_thresholds:
        raise RuntimeError("Btag thresholds for period '{}' are not found.".format(period))
    if wp not in btag_thresholds[period]:
        raise RuntimeError("Btag working point '{}' for period '{}' is not found.".format(wp, period))
    return btag_thresholds[period][wp]

def getMetFilters(period, isMC):
    """ Returns list of suggested MET filters that should be applied in event-by-event basis.
        The recommended values are taken from https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2
    """

    metFilters_common = [
        "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter",
        "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter"
    ]
    filters = metFilters_common[:]
    if period in ['Run2017', 'Run2018', 'Run2018ABC', 'Run2018D']:
        filters.append("ecalBadCalibReducedMINIAODFilter")
    if not isMC:
        filters.append("Flag_eeBadScFilter")
    return filters

def getReportInterval(maxEvents):
    """ Return reporting interval based on maximal number of events to process """
    if maxEvents > 10000 or maxEvents <= 0:
        return 1000
    if maxEvents > 10:
        return maxEvents // 10
    return 1
