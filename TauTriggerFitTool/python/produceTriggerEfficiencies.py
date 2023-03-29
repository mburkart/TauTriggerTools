from ROOT import *
from array import array
gROOT.SetBatch(True)
from math import sqrt
from functions import *
from binning2017 import *
from binning2018 import *
import numpy as np

#choose which year do you want to run it:
Samples2016 = False
Samples2017 = False
Samples2018 = True


# NTuples produced for VVLoose WP using 2017v2 MVA
files2016 = ("/afs/cern.ch/user/h/hsert/public/Run2SamplesTrigger/Ntuple_SingleMuon_Run2016BtoH-17Jul2018_190228_SSsubtraction_VVLooseWP2017v2_forFit.root", "/afs/cern.ch/user/h/hsert/public/Run2SamplesTrigger/Ntuple_DYJets_RunIISummer16MiniAODv3_94X_mcRun2_ext1ANDext2-v2_190306_PUreweight_OStauGenMatched_VVLooseWP2017v2_forFit.root")

files2017 = ("/afs/cern.ch/user/h/hsert/public/Fall17Samples_31MarData_12AprMC/NTuple_Data_Run2017BCDEF_31Mar2018_SSsubtraction_VVLooseWP2017v2.root", "/afs/cern.ch/user/h/hsert/public/Fall17Samples_31MarData_12AprMC/NTuple_DYJetsToLL_12Apr2018_v1Andext1v1_12062018_puWeightsANDtauEScorrectionIncluded_OStauGenMatched_VVLooseWP2017v2.root")

files2018 = ("/afs/cern.ch/user/h/hsert/public/Run2SamplesTrigger/Ntuple_SingleMuon_Run2018ABCDReReco17SepPromptRecoD_190121_SSsubtraction_VVLooseWP2017v2_forFit.root","/afs/cern.ch/user/h/hsert/public/Run2SamplesTrigger/Ntuple_DYJetsToLL_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_190121_PUreweight1000MCbin_OStauGenMatched_VVLooseWP2017v2_forFit.root")

if(Samples2016 and not Samples2018 and not Samples2017):
	files = files2016
elif(Samples2017 and not Samples2018 and not Samples2016):
	files = files2017
elif(Samples2018 and not Samples2017 and not Samples2016):
	files = files2018
else:
	"Please select only one of the year!"

gStyle.SetFrameLineWidth(1)
gStyle.SetPadBottomMargin(0.13)
gStyle.SetPadLeftMargin(0.15)
gStyle.SetPadTopMargin(0.09)
gStyle.SetPadRightMargin(0.05)

triggers = ["ditau", "mutau", "etau"]
types = ["DATA", "MC"]
WPs = ["vvlooseTauMVA", "vlooseTauMVA", "looseTauMVA", "mediumTauMVA", "tightTauMVA", "vtightTauMVA", "vvtightTauMVA"]
tauDMs = ["dm0", "dm1", "dm10"]

isDMspesific = True

if(Samples2016):
	outputname =  "tauTriggerEfficiencies2016.root"
elif(Samples2018):
	outputname =  "tauTriggerEfficiencies2018.root"
else:
	outputname = "tauTriggerEfficiencies2017_final_etauESshift_version2.root"


# get binning from binning2017.py and binning2018.py files
if(Samples2016):
	bins = getbinning2017()
elif(Samples2017):
	bins = getbinning2017()
elif(Samples2018):
	bins = getbinning2018()

#bin = bins.getBinning()
#binDM = bins.getBinningDM()
bin_ref = np.arange(20, 501, 1).tolist()
bin = {}
binDM = {}
for trig in triggers:
	bin[trig] = bin_ref
	binDM[trig] = {}
	for dm in tauDMs:
		binDM[trig][dm] = {}
		for sample_type in types:
			binDM[trig][dm][sample_type] = bin_ref
#binDM = bins.getBinningPerDM()

hPtDen = [[],[],[]]
hPtNum = [[],[],[]]

hPtDenDM = [[],[],[],[]]
hPtNumDM = [[],[],[],[]]

for ipath, trigger in enumerate(triggers):

	print "bin[", trigger, "]", bin[trigger]
	hPtNum.append([])
	hPtDen.append([])
	# per DM
	hPtNumDM.append([])
	hPtDenDM.append([])

	for index, typ in enumerate(types):
		hPtNum[ipath].append([])
		hPtDen[ipath].append([])
		# per DM
		hPtNumDM[ipath].append([])
		hPtDenDM[ipath].append([])

		for ind, wp in enumerate(WPs):
			histoname = "histo_" + typ + "_" + wp + "_" + trigger
			hPtNumDM[ipath][index].append([])
			hPtDenDM[ipath][index].append([])

			hPtDen[ipath][index].append(TH1F (histoname + "_Den", "", len(bin[trigger])-1, array('f',bin[trigger])))
			hPtNum[ipath][index].append(TH1F (histoname + "_Num", "", len(bin[trigger])-1, array('f',bin[trigger])))

			# per DM
			for idm, DM in enumerate(tauDMs):
				hPtDenDM[ipath][index][ind].append(TH1F (histoname + "_Den_" + DM, "", len(binDM[trigger][DM][typ])-1, array('f',binDM[trigger][DM][typ])))
				hPtNumDM[ipath][index][ind].append(TH1F (histoname + "_Num_" + DM, "", len(binDM[trigger][DM][typ])-1, array('f',binDM[trigger][DM][typ])))

for index, filename in enumerate(files):
	print  "filename", filename

	file = TFile.Open(filename)
	tree = file.Get('TagAndProbe')
	triggerNamesTree = file.Get("triggerNames")

	print "Populating histograms"

	Nevts = 0
	for iEv in range (0, tree.GetEntries()):
		tree.GetEntry(iEv)

		tauPt = tree.tauPt
		HLTPt = tree.hltPt
		tauEta = tree.tauEta
		tauPhi = tree.tauPhi
		tauDM = tree.tauDM
		Nvtx = tree.Nvtx

		# tau Energy Shift (SF) is only applied for 2017 for now!
		if("DYJets" in filename):
			puweight = tree.puweight
			if (Samples2017):
				tauPt_ESshifted = tree.tauPt_ESshifted
			else:
				tauPt_ESshifted = tree.tauPt
		else:
			puweight=1
			tauPt_ESshifted = tree.tauPt

		vvlooseWP = tree.byVVLooseIsolationMVArun2017v2DBoldDMwLT2017
		vlooseWP = tree.byVLooseIsolationMVArun2017v2DBoldDMwLT2017
		looseWP = tree.byLooseIsolationMVArun2017v2DBoldDMwLT2017
		mediumWP = tree.byMediumIsolationMVArun2017v2DBoldDMwLT2017
		tightWP = tree.byTightIsolationMVArun2017v2DBoldDMwLT2017
		vtightWP = tree.byVTightIsolationMVArun2017v2DBoldDMwLT2017
		vvtightWP = tree.byVVTightIsolationMVArun2017v2DBoldDMwLT2017

		if("Run2016BtoH" in filename or "190306" in filename):
			hasHLTditauPath_3or4 = tree.hasHLTditauPath_3or4
			hasHLTetauPath_0and1 = tree.hasHLTetauPath_0and1
			hasHLTmutauPath_1 = tree.hasHLTmutauPath_1
		elif("Run2017B" in filename or "12062018" in filename):
			hasHLTditauPath_9or10or11 = tree.hasHLTditauPath_9or10or11
			hasHLTetauPath_13 = tree.hasHLTetau_Path_13
			hasHLTmutauPath_13 = tree.hasHLTmutauPath_13
		else:
			hasHLTditauPath_4or5or6noHPS = tree.hasHLTditauPath_4or5or6noHPS
			hasHLTetauPath_8noHPS = tree.hasHLTetauPath_8noHPS
			hasHLTmutauPath_8noHPS = tree.hasHLTmutauPath_8noHPS
			hasHLTditauPath_15or20HPS = tree.hasHLTditauPath_15or20HPS
			hasHLTetauPath_14HPS = tree.hasHLTetauPath_14HPS
			hasHLTmutauPath_14HPS = tree.hasHLTmutauPath_14HPS

		Nevents = tree.EventNumber
		Nevts =Nevts + 1

		#bkgSubW = 1. if tree.isOS else -1.
		weight = tree.bkgSubW*puweight

		if("Run2018A" in filename or "Autumn18" in filename):
                        HLTHPSpaths18 = [hasHLTditauPath_15or20HPS, hasHLTmutauPath_14HPS , hasHLTetauPath_14HPS ]
			HLTpaths18 = [hasHLTditauPath_4or5or6noHPS, hasHLTmutauPath_8noHPS, hasHLTetauPath_8noHPS]
		elif("Run2017B" in filename or "12062018" in filename):
			HLTpaths17 = [hasHLTditauPath_9or10or11, hasHLTmutauPath_13 , hasHLTetauPath_13 ]
		else:
			HLTpaths16 = [hasHLTditauPath_3or4, hasHLTmutauPath_1 , hasHLTetauPath_0and1 ]

		WPoints = [vvlooseWP, vlooseWP, looseWP, mediumWP, tightWP, vtightWP, vvtightWP]

		oneProng = False
		oneProngPiZero= False
		threeProng = False

		if (tauDM ==0):
			oneProng = True
		if (tauDM ==1):
			oneProngPiZero = True
		if (tauDM ==10):
			threeProng = True

		DMs = [oneProng, oneProngPiZero, threeProng]

		# Filling the histograms
		for WPind, WP in enumerate(WPoints):
			if((WP > 0) and ("Run2016BtoH" in filename or "190306" in filename)):
				for ipath, trigger in enumerate(HLTpaths16):
					hPtDen[ipath][index][WPind].Fill(tauPt_ESshifted, weight)
					for idm, DM in enumerate(DMs):
						if(DM == True):
							hPtDenDM[ipath][index][WPind][idm].Fill(tauPt_ESshifted, weight)
				if ( HLTpaths16[0] >0 ):
					hPtNum[0][index][WPind].Fill(tauPt_ESshifted, weight)
					for idm, DM in enumerate(DMs):
						if(DM == True):
							hPtNumDM[0][index][WPind][idm].Fill(tauPt_ESshifted, weight)
				if( HLTpaths16[1] >0 ):
					hPtNum[1][index][WPind].Fill(tauPt_ESshifted, weight)
					for idm, DM in enumerate(DMs):
						if(DM == True):
							hPtNumDM[1][index][WPind][idm].Fill(tauPt_ESshifted, weight)
				if( HLTpaths16[2] >0 ):
					hPtNum[2][index][WPind].Fill(tauPt_ESshifted, weight)
					for idm, DM in enumerate(DMs):
						if(DM == True):
							hPtNumDM[2][index][WPind][idm].Fill(tauPt_ESshifted, weight)

			elif((WP > 0) and ("Run2017B" in filename or "12062018" in filename)):
				for ipath, trigger in enumerate(HLTpaths17):
					hPtDen[ipath][index][WPind].Fill(tauPt_ESshifted, weight)
					for idm, DM in enumerate(DMs):
						if(DM == True):
							hPtDenDM[ipath][index][WPind][idm].Fill(tauPt_ESshifted, weight)
				if ( HLTpaths17[0] >0 ):
					hPtNum[0][index][WPind].Fill(tauPt_ESshifted, weight)
					for idm, DM in enumerate(DMs):
						if(DM == True):
							hPtNumDM[0][index][WPind][idm].Fill(tauPt_ESshifted, weight)
				if( HLTpaths17[1] >0 ):
					hPtNum[1][index][WPind].Fill(tauPt_ESshifted, weight)
					for idm, DM in enumerate(DMs):
						if(DM == True):
							hPtNumDM[1][index][WPind][idm].Fill(tauPt_ESshifted, weight)
				if( HLTpaths17[2] >0 ):
					hPtNum[2][index][WPind].Fill(tauPt_ESshifted, weight)
					for idm, DM in enumerate(DMs):
						if(DM == True):
							hPtNumDM[2][index][WPind][idm].Fill(tauPt_ESshifted, weight)

			elif((WP > 0) and ("Run2018A" in filename or "Autumn18" in filename)):
				if("Run2018A" in filename):
					if(tree.RunNumber < 317509):
						for ipath, trigger in enumerate(HLTpaths18):
							hPtDen[ipath][index][WPind].Fill(tauPt_ESshifted, weight)
							for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtDenDM[ipath][index][WPind][idm].Fill(tauPt_ESshifted, weight)
						if ( HLTpaths18[0] >0 ):
							hPtNum[0][index][WPind].Fill(tauPt_ESshifted, weight)
							for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtNumDM[0][index][WPind][idm].Fill(tauPt_ESshifted, weight)
						if( HLTpaths18[1] >0 ):
							hPtNum[1][index][WPind].Fill(tauPt_ESshifted, weight)
							for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtNumDM[1][index][WPind][idm].Fill(tauPt_ESshifted, weight)
						if( HLTpaths18[2] >0 ):
							hPtNum[2][index][WPind].Fill(tauPt_ESshifted, weight)
							for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtNumDM[2][index][WPind][idm].Fill(tauPt_ESshifted, weight)
					else:
						for ipath, trigger in enumerate(HLTHPSpaths18):
							hPtDen[ipath][index][WPind].Fill(tauPt_ESshifted, weight)
							for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtDenDM[ipath][index][WPind][idm].Fill(tauPt_ESshifted, weight)
						if ( HLTHPSpaths18[0] >0 ):
							hPtNum[0][index][WPind].Fill(tauPt_ESshifted, weight)
							for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtNumDM[0][index][WPind][idm].Fill(tauPt_ESshifted, weight)
						if( HLTHPSpaths18[1] >0 ):
							hPtNum[1][index][WPind].Fill(tauPt_ESshifted, weight)
							for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtNumDM[1][index][WPind][idm].Fill(tauPt_ESshifted, weight)
						if( HLTHPSpaths18[2] >0 ):
							hPtNum[2][index][WPind].Fill(tauPt_ESshifted, weight)
							for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtNumDM[2][index][WPind][idm].Fill(tauPt_ESshifted, weight)

				elif((WP > 0) and ("Autumn18" in filename)):

					for ipath, trigger in enumerate(HLTHPSpaths18):
						hPtDen[ipath][index][WPind].Fill(tauPt_ESshifted, weight)
						for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtDenDM[ipath][index][WPind][idm].Fill(tauPt_ESshifted, weight)
					if ( HLTHPSpaths18[0] >0 ):
						hPtNum[0][index][WPind].Fill(tauPt_ESshifted, weight)
						for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtNumDM[0][index][WPind][idm].Fill(tauPt_ESshifted, weight)
					if( HLTHPSpaths18[1] >0 ):
						hPtNum[1][index][WPind].Fill(tauPt_ESshifted, weight)
						for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtNumDM[1][index][WPind][idm].Fill(tauPt_ESshifted, weight)
					if( HLTHPSpaths18[2] >0 ):
						hPtNum[2][index][WPind].Fill(tauPt_ESshifted, weight)
						for idm, DM in enumerate(DMs):
								if(DM == True):
									hPtNumDM[2][index][WPind][idm].Fill(tauPt_ESshifted, weight)


file = TFile( "../data/"+outputname, 'recreate')

# check bin contents of the histograms to avoid cases Num > Den, triggered due to SS subtraction (negative weights)
for ipath, trigger in enumerate(triggers):
	for WPind, wp in enumerate(WPs):
		for index, typ in enumerate(types):

			nbins = hPtNum[ipath][index][WPind].GetNbinsX()
			for binid in range(0, nbins + 1):
				if(hPtNum[ipath][index][WPind].GetBinContent(binid) > hPtDen[ipath][index][WPind].GetBinContent(binid)):
					hPtNum[ipath][index][WPind].SetBinContent(binid, hPtDen[ipath][index][WPind].GetBinContent(binid))

			for idm, DM in enumerate(tauDMs):
				for binid in range(0, nbins + 1):
					if(hPtNumDM[ipath][index][WPind][idm].GetBinContent(binid) > hPtDenDM[ipath][index][WPind][idm].GetBinContent(binid)):
						hPtNumDM[ipath][index][WPind][idm].SetBinContent(binid, hPtDenDM[ipath][index][WPind][idm].GetBinContent(binid))

# efficiency calculation after filling the histograms for 3 different triggers for each WPs of DATA and MC
for ipath, trigger in enumerate(triggers):

	for WPind, wp in enumerate(WPs):

		for index, typ in enumerate(types):

			g_efficiency =TGraphAsymmErrors()
			g_efficiency.BayesDivide(hPtNum[ipath][index][WPind],hPtDen[ipath][index][WPind])

			funct = functions(g_efficiency, trigger + "Efficiency_" + wp +"_"+ typ, 0, 0, 0, 0, 0, 0, 0)
			h_efficiency = funct.getTH1FfromTGraphAsymmErrors()

			# write the histograms/graphs into the output ROOT file before the fit
			g_efficiency.Write(trigger +"_gEfficiency_" + wp +"_"+ typ)
			h_efficiency.Write(trigger +"_hEfficiency_" + wp +"_"+ typ)
			hPtNum[ipath][index][WPind].Write(trigger +"_pass_" + wp +"_"+ typ)
			hPtDen[ipath][index][WPind].Write(trigger +"_total_" + wp +"_"+ typ)

			# Set the title of the histograms/graphs and their axes
			g_efficiency.SetTitle(trigger +"Path_" + wp +"_"+ typ)
			g_efficiency.GetYaxis().SetTitle("Efficiency")
			g_efficiency.GetXaxis().SetTitle("Offline p_{T}^{#tau} [GeV]")
			h_efficiency.SetTitle(trigger +"Path_" + wp +"_"+ typ)
			h_efficiency.GetYaxis().SetTitle("Efficiency")
			h_efficiency.GetXaxis().SetTitle("Offline p_{T}^{#tau} [GeV]")

			# per DM efficiencies
			for idm, DM in enumerate(tauDMs):
				g_efficiencyDM = TGraphAsymmErrors()
				g_efficiencyDM.BayesDivide(hPtNumDM[ipath][index][WPind][idm],hPtDenDM[ipath][index][WPind][idm])
				funct2 = functions(g_efficiencyDM, trigger + "_Efficiency" + wp +"_"+ typ + "_" + DM, idm, 0 ,0, 0, 0, 0, 0)
				h_efficiencyDM = funct2.getTH1FfromTGraphAsymmErrors()
				g_efficiencyDM.Write(trigger +"_gEfficiency_" + wp +"_" + typ + "_" + DM)
				h_efficiencyDM.Write(trigger +"_hEfficiency_" + wp +"_" + typ + "_" + DM)
				hPtNumDM[ipath][index][WPind][idm].Write(trigger +"_pass_" + wp +"_" + typ + "_" + DM)
				hPtDenDM[ipath][index][WPind][idm].Write(trigger +"_total_" + wp +"_" + typ + "_" + DM)

file.Close()
print "The output ROOT file has been created: ../data/" + outputname
