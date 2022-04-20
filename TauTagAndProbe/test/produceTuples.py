import os
import re
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from TauTriggerTools.Common.ProduceHelpers import *

options = VarParsing('analysis')
options.register('inputFileList', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Text file with a list of the input root files to process.")
options.register('fileNamePrefix', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Prefix to add to input file names. Use file: for the files in the local file system.")
options.register('outputTupleFile', 'eventTuple.root', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Event tuple file.")
options.register('skipEvents', -1, VarParsing.multiplicity.singleton, VarParsing.varType.int,
                 "Number of events to skip")
options.register('eventList', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "List of events to process.")
options.register('lumiFile', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "JSON file with lumi mask.")
options.register('period', 'Run2018', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Data taking period")
options.register('triggerProcess', 'HLT', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Trigger process")
options.register('metFiltersProcess', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Process for MET filters. If empty, it will be deduced based on the period.")
options.register('globalTag', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Global tag. If empty, it will be deduced based on the period.")
options.register('isMC', True, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "MC or Data")
options.register('runDeepTau', True, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "Run DeepTau IDs")
options.register('relVal', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Minimal setup for RelVal")
options.register('pureGenMode', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Don't apply any offline selection or tagging.")
options.register('wantSummary', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Print run summary at the end of the job.")
options.parseArguments()

processName = "TagAndProbe"
process = cms.Process(processName)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

if len(options.globalTag) == 0:
    process.GlobalTag.globaltag = getGlobalTag(options.period, options.isMC)
else:
    process.GlobalTag.globaltag = options.globalTag
process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring())
process.TFileService = cms.Service('TFileService', fileName=cms.string(options.outputTupleFile))

if len(options.inputFileList) > 0:
    readFileList(process.source.fileNames, options.inputFileList, options.fileNamePrefix)
elif len(options.inputFiles) > 0:
    addFilesToList(process.source.fileNames, options.inputFiles, options.fileNamePrefix)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
if options.maxEvents > 0:
    process.maxEvents.input = cms.untracked.int32(options.maxEvents)
if options.skipEvents > 0:
    process.source.skipEvents = cms.untracked.uint32(options.skipEvents)
if len(options.eventList) > 0:
    process.source.eventsToProcess = cms.untracked.VEventRange(options.eventList.split(','))
if len(options.lumiFile) > 0:
    import FWCore.PythonUtilities.LumiList as LumiList
    process.source.lumisToProcess = LumiList.LumiList(filename = options.lumiFile).getVLuminosityBlockRange()

year = getYear(options.period)

# Update electron ID according recommendations from https://twiki.cern.ch/twiki/bin/view/CMS/EgammaMiniAODV2
if options.relVal or options.pureGenMode:
    process.egammaPostRecoSeq = cms.Sequence()
else:
    from RecoEgamma.EgammaTools.EgammaPostRecoTools import setupEgammaPostRecoSeq
    ele_era = {
        2016: '2016-Legacy',
        2017: '2017-Nov17ReReco',
        2018: '2018-Prompt'
    }
    setupEgammaPostRecoSeq(process, runVID=True, runEnergyCorrections=False, era=ele_era[year])

# Update tau IDs according recommendations from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuidePFTauID
process.tauSequence = cms.Sequence()
if options.relVal:
    tauSrc_InputTag = cms.InputTag("slimmedTaus")
else:
    import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
    updatedTauName = "slimmedTausNewID"
    tauIdsToKeep = [ "2017v2" ]
    if options.runDeepTau:
        tauIdsToKeep.append("deepTau2017v2p1")
    tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, cms, debug=False, updatedTauName=updatedTauName,
                                              toKeep=tauIdsToKeep)
    tauIdEmbedder.runTauID()
    process.tauSequence += process.rerunMvaIsolationSequence
    process.tauSequence += getattr(process, updatedTauName)
    tauSrc_InputTag = cms.InputTag(updatedTauName)


# Update MET filters according recommendations from https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2
# Using post-Moriond2019 (a more complete) list of noisy crystals
process.metFilterSequence = cms.Sequence()
customMetFilters = cms.PSet()
if not (options.pureGenMode or options.relVal) and year in [ 2017, 2018 ]:
    process.load('RecoMET.METFilters.ecalBadCalibFilter_cfi')
    baddetEcallist = cms.vuint32([
        872439604,872422825,872420274,872423218,872423215,872416066,872435036,872439336,
        872420273,872436907,872420147,872439731,872436657,872420397,872439732,872439339,
        872439603,872422436,872439861,872437051,872437052,872420649,872421950,872437185,
        872422564,872421566,872421695,872421955,872421567,872437184,872421951,872421694,
        872437056,872437057,872437313,872438182,872438951,872439990,872439864,872439609,
        872437181,872437182,872437053,872436794,872436667,872436536,872421541,872421413,
        872421414,872421031,872423083,872421439])
    process.ecalBadCalibReducedMINIAODFilter = cms.EDFilter("EcalBadCalibFilter",
        EcalRecHitSource = cms.InputTag("reducedEgamma:reducedEERecHits"),
        ecalMinEt = cms.double(50.),
        baddetEcal = baddetEcallist,
        taggingMode = cms.bool(True),
        debug = cms.bool(False)
    )
    process.metFilterSequence += process.ecalBadCalibReducedMINIAODFilter
    customMetFilters.ecalBadCalibReducedMINIAODFilter = cms.InputTag("ecalBadCalibReducedMINIAODFilter")

if len(options.metFiltersProcess) == 0:
    metFiltersProcess = 'PAT'
    if year in [ 2016, 2018 ] and not options.isMC:
        metFiltersProcess = 'RECO'
else:
    metFiltersProcess = options.metFiltersProcess

# Re-apply MET corrections
process.metSequence = cms.Sequence()
if not (options.pureGenMode or options.relVal) and options.period in [ 'Run2016', 'Run2017' ]:
    met_run_params = { }
    if options.period == 'Run2017':
        met_run_params = {
            'fixEE2017': True,
            'fixEE2017Params': {
                'userawPt': True,
                'ptThreshold':50.0,
                'minEtaThreshold':2.65,
                'maxEtaThreshold': 3.139
            }
        }
    from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
    runMetCorAndUncFromMiniAOD(process, isData = not options.isMC, postfix="Updated", **met_run_params)
    metInputTag = cms.InputTag('slimmedMETsUpdated', '', processName)
    process.metSequence += process.fullPatMetSequenceUpdated
else:
    metInputTag = cms.InputTag('slimmedMETs')

from TauTriggerTools.Common import TriggerConfig
trigFile = '{}/src/TauTriggerTools/TauTagAndProbe/data/{}/triggers.json'.format(os.environ['CMSSW_BASE'], year)
hltPaths, tagHltPaths = TriggerConfig.LoadAsVPSet(trigFile)

import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt

process.hltFilter = hlt.hltHighLevel.clone(
    TriggerResultsTag = cms.InputTag("TriggerResults", "", "HLT"),
    HLTPaths = [p + '*' for p in tagHltPaths],
    andOr = cms.bool(True), # True (OR) accept if ANY is true, False (AND) accept if ALL are true
    throw = cms.bool(True) # if True: throws exception if a trigger path is invalid
)

process.patTriggerUnpacker = cms.EDProducer("PATTriggerObjectStandAloneUnpacker",
    patTriggerObjectsStandAlone = cms.InputTag("slimmedPatTrigger"),
    triggerResults              = cms.InputTag('TriggerResults', '', options.triggerProcess),
    unpackFilterLabels          = cms.bool(True)
)

process.selectionFilter = cms.EDFilter("TauTriggerSelectionFilter",
    enabled           = cms.bool(not options.pureGenMode),
    electrons         = cms.InputTag('slimmedElectrons'),
    muons             = cms.InputTag('slimmedMuons'),
    jets              = cms.InputTag('slimmedJets'),
    met               = metInputTag,
    metFiltersResults = cms.InputTag('TriggerResults', '', metFiltersProcess),
    customMetFilters  = customMetFilters,
    btagThreshold     = cms.double(-1),
    metFilters        = cms.vstring(getMetFilters(options.period, options.isMC)),
    mtCut             = cms.double(-1)
)

process.summaryProducer = cms.EDProducer("TauTriggerSummaryTupleProducer",
    isMC            = cms.bool(options.isMC),
    genEvent        = cms.InputTag('generator'),
    puInfo          = cms.InputTag('slimmedAddPileupInfo'),
    vertices        = cms.InputTag('offlineSlimmedPrimaryVertices'),
    hltPaths        = hltPaths
)

process.tupleProducer = cms.EDProducer("TauTriggerTupleProducer",
    isMC            = cms.bool(options.isMC),
    genEvent        = cms.InputTag('generator'),
    puInfo          = cms.InputTag('slimmedAddPileupInfo'),
    genParticles    = cms.InputTag('prunedGenParticles'),
    vertices        = cms.InputTag('offlineSlimmedPrimaryVertices'),
    signalMuon      = cms.InputTag('selectionFilter'),
    taus            = tauSrc_InputTag,
    jets            = cms.InputTag('slimmedJets'),
    met             = metInputTag,
    btagThreshold   = cms.double(getBtagThreshold(options.period, 'Loose')),
    hltPaths        = hltPaths,
    triggerProcess  = cms.string(options.triggerProcess),
    triggerObjects  = cms.InputTag('patTriggerUnpacker'),
    l1Taus          = cms.InputTag("caloStage2Digis", "Tau", "RECO")
)

process.p = cms.Path(
    process.summaryProducer +
    process.hltFilter +
    process.egammaPostRecoSeq +
    process.metSequence +
    process.metFilterSequence +
    process.selectionFilter +
    process.tauSequence +
    process.patTriggerUnpacker +
    process.tupleProducer
)

# Verbosity customization
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = getReportInterval(process.maxEvents.input.value())
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(options.wantSummary) )
