from array import array
import argparse
from math import sqrt

import ROOT as root

root.PyConfig.IgnoreCommandLineOptions = True
root.gROOT.SetBatch(True)

from functions import *
from fittingTool import fittingTool

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input file")
parser.add_argument("-o", "--output", help="Output file")
parser.add_argument("-t", "--triggers", nargs="*", default=["etau", "mutau", "ditau"],
                    help="Trigger to be considered.")
parser.add_argument("-w", "--working-points", nargs="*",
                    default=["vloose", "loose", "medium", "tight", "vtight", "vvtight"],
                    help="Working points to be considered")
parser.add_argument("-d", "--data", nargs="*", default=["DATA", "MC", "EMB"])
parser.add_argument("--mva", action="store_true",
                    help="Use MVA tau id instead of DeepTau ID.")
args = parser.parse_args()


# filename = "../data/output_2017_tau_leg_earlyFilters.root"
filename = args.input
file = root.TFile.Open(filename, "r")


triggers = args.triggers
types = args.data
if args.mva:
    WPs = [wp + "TauMVA" for wp in args.working_points]
    tauDMs = ["dm0", "dm1", "dm10"]
else:
    WPs = [wp + "DeepTau" for wp in args.working_points]
    tauDMs = ["dm0", "dm1", "dm10", "dm11"]

isDMspesific = True

root.gStyle.SetFrameLineWidth(1)
root.gStyle.SetPadBottomMargin(0.13)
root.gStyle.SetPadLeftMargin(0.15)
root.gStyle.SetPadTopMargin(0.09)
root.gStyle.SetPadRightMargin(0.05)

# outputfile = TFile( "../data/"+outputname, 'recreate')
outputfile = root.TFile(args.output, 'recreate')

nfailed = 0
# efficiency calculation after filling the histograms for 3 different triggers for each WPs of DATA and MC
for ipath, trigger in enumerate(triggers):
	for WPind, wp in enumerate(WPs):
		f1 =[]
		h_errBandDM68 = [[], []]
		g_errBandDM68 = [[], []]
		for idm, DM in enumerate(tauDMs):
			h_errBandDM68.append([])
			g_errBandDM68.append([])
			for index, typ in enumerate(types):
				histoname = "histo_" + typ + "_" + wp + "_" + trigger
				h_errBandDM68[idm].append(root.TH1F(histoname+ "_" + DM +"_CL68","histo of 0.68 confidence band", 480, 20, 500))
				g_errBandDM68[idm].append(root.TGraphErrors())
		
		for index, typ in enumerate(types):
			
			f1.append(root.TF1( 'f1'+typ, '[5] - ROOT::Math::crystalball_cdf(-x, [0], [1], [2], [3])*([4])'))
			if(index ==0):
				f1[index].SetLineColor( root.kBlue)
			else:
				f1[index].SetLineColor( root.kRed)
			f1[index].SetParName( 0, "alpha" )
			f1[index].SetParName( 1, "n" )
			f1[index].SetParName( 2, "sigma" )
			f1[index].SetParName( 3, "x0" )
			f1[index].SetParName( 4, "scale" )
			f1[index].SetParName( 5, "y-rise" )

		f2 = [[],[]]
                for idm, DM in enumerate(tauDMs):
                        f2.append([])
                        for index, typ in enumerate(types):
                                f2[idm].append(root.TF1( 'f2_'+ DM  +"_" + typ, '[5] - ROOT::Math::crystalball_cdf(-x, [0], [1], [2], [3])*([4])'))
                                if(idm ==0): f2[idm][index].SetLineColor( root.kBlue )
                                elif(idm ==1): f2[idm][index].SetLineColor( root.kRed )
                                elif(idm ==2): f2[idm][index].SetLineColor( root.kGreen+3 )
                                if( isDMspesific):
                                        if index==0: f2[idm][0].SetLineColor( root.kBlue )
                                        elif index==1: f2[idm][1].SetLineColor( root.kRed )
                                f2[idm][index].SetParName( 0, "alpha" )
                                f2[idm][index].SetParName( 1, "n" )
                                f2[idm][index].SetParName( 2, "sigma" )
                                f2[idm][index].SetParName( 3, "x0" )
                                f2[idm][index].SetParName( 4, "scale" )
                                f2[idm][index].SetParName( 5, "y-rise" )

		g_errBand68 = []
		h_errBand68 = []
		
		for index, typ in enumerate(types):
	
			print "Fit is performed for", trigger, "trigger in", wp ,"WP for", typ
			
			gEfficiency = root.TGraphAsymmErrors()
                        query = "graphv2_" + trigger +"TriggerEfficiency_" + wp +"_"+ typ
                        print "query: ", query
			gEfficiency = file.Get(query)
                        if isinstance(gEfficiency, root.TObject) and not isinstance(gEfficiency, root.RooHist):
                            print "Histogram {} not found in file {}".format(query, args.input)
                            raise Exception

			fitter = fittingTool(f1[index], gEfficiency)

			print "trigger", trigger

			h_errBand68.append(root.TH1F(histoname+"_CL68","histo of 0.68 confidence band", 480, 20, 500))
			g_errBand68.append(root.TGraphErrors())
			
			fit_result = fitter.performFit()
			if int(fit_result) != 0: nfailed += 1
			
			funct = functions(gEfficiency, "histo_" + trigger + "ErrorBand_" + wp +"_"+ typ, idm, index, f1, h_errBand68[index], g_errBand68[index], fit_result, 0.68)
			
			h_errBand68[index], g_errBand68[index] = funct.getConfidenceInterval() 

			# Set the title of the histograms/graphs and their axes
			gEfficiency.SetTitle(trigger +"Path_" + wp +"_"+ typ)
			gEfficiency.GetYaxis().SetTitle("Efficiency")
			gEfficiency.GetXaxis().SetTitle("Offline p_{T}^{#tau} [GeV]")

			h_errBand68[index].SetTitle(trigger +"Path_" + wp +"_"+ typ)
			h_errBand68[index].GetYaxis().SetTitle("Efficiency")
			h_errBand68[index].GetXaxis().SetTitle("Offline p_{T}^{#tau} [GeV]")
			g_errBand68[index].SetTitle(trigger +"Path_" + wp +"_"+ typ)
			g_errBand68[index].GetYaxis().SetTitle("Efficiency")
			g_errBand68[index].GetXaxis().SetTitle("Offline p_{T}^{#tau} [GeV]")
			
			# write the histograms/graphs into the output ROOT file after the fit
			gEfficiency.Write(trigger +"_gEffi_" + wp +"_"+ typ)
			h_errBand68[index].Write(trigger  + "_herrBand_"  + wp +"_"+ typ)
			g_errBand68[index].Write(trigger + "_gerrBand_"+ wp +"_"+ typ)
	
			#======== Relative error of the fit: "fit +/- error/ fit " ===================
			relativeErrorUP = root.TGraphAsymmErrors()
			relativeErrorDown = root.TGraphAsymmErrors()
			relativeErrorUP, relativeErrorDown = funct.createRelativeErrors()

			relativeErrorUP.Write(trigger  +"_relativeErrorUp_" + wp + "_"+ typ )
			relativeErrorDown.Write(trigger + "_relativeErrorDown_"+   wp +"_"+ typ	)
	
			# per DM efficiencies
			for idm, DM in enumerate(tauDMs):
				print "Fit is performed for", trigger, "trigger in", wp ,"WP for", typ , " per DM", DM
				
				gEfficiencyDM = root.TGraphAsymmErrors()
                                query = "graphv2_" + trigger + "TriggerEfficiency_" + wp +"_"+ DM +"_"+ typ
                                print "query: ", query
				gEfficiencyDM = file.Get(query)
                                if isinstance(gEfficiency, root.TObject) and not isinstance(gEfficiency, root.RooHist):
                                    print "Histogram {} not found in file {}".format(query, args.input)
                                    raise Exception
				
				fitter = fittingTool(f2[idm][index], gEfficiencyDM)

				fit_result2 = fitter.performFit()

				if int(fit_result2) != 0: nfailed += 1			
	
				functDM = functions(gEfficiency, "histo_" + trigger + "_" + wp +"_"+ typ, idm, index, f2[idm] , h_errBandDM68[idm][index], g_errBandDM68[idm][index], fit_result2, 0.68)
				h_errBandDM68[idm][index], g_errBandDM68[idm][index] = functDM.getConfidenceInterval()

				f2result = root.TF1( 'f2', '[5] - ROOT::Math::crystalball_cdf(-x, [0], [1], [2], [3])*([4])', 0, 500)
				f2result.SetParameter(0, f2[idm][index].GetParameter(0))
				f2result.SetParameter(1, f2[idm][index].GetParameter(1))
				f2result.SetParameter(2, f2[idm][index].GetParameter(2))
				f2result.SetParameter(3, f2[idm][index].GetParameter(3))
				f2result.SetParameter(4, f2[idm][index].GetParameter(4))
				f2result.SetParameter(5, f2[idm][index].GetParameter(5))


				gEfficiencyDM.Write(trigger +"_gEffi_"+ wp  +"_"+ DM +"_" + typ)
				f2result.Write(trigger +"_fit_"+ wp +"_"+ DM +"_" + typ)
				h_errBandDM68[idm][index].Write(trigger  + "_herrband_" + wp  +"_"+ DM +"_" + typ)
				g_errBandDM68[idm][index].Write(trigger  + "_gerrband_" + wp  +"_"+ DM +"_" + typ)


		# Getting Scale Factors
		SF = root.TGraphAsymmErrors()
		SFerror = root.TH1F()
		functSF = functions(gEfficiency, "histo_" + trigger + "ErrorBand_" + wp +"_"+ typ, idm, index, f1, h_errBand68, g_errBand68, fit_result, 0.68) 
			
		SFerror = functSF.getScaleFactorError()
		SF = functSF.getScaleFactor()
		SF.Write(trigger + '_ScaleFactor_' + wp)
		SFerror.Write(trigger + '_ScaleFactorUnc_' + wp)
		
		# Getting Scale Factors per decay mode
		for idm, DM in enumerate(tauDMs):
			SFdm = root.TGraphErrors()
			SFdmerror = root.TH1F()
			functDMSF = functions(gEfficiency, "histo_" + trigger + "ErrorBand_" + wp +"_"+ typ, idm, index, f2[idm], h_errBandDM68[idm], g_errBandDM68[idm], fit_result2, 0.68) 
			
			SFdmerror = functDMSF.getScaleFactorError()
			SFdm = functDMSF.getScaleFactor()
			SFdm.Write(trigger + "_ScaleFactor_" + wp + '_' + DM)
			SFdmerror.Write(trigger + '_ScaleFactorUnc_' + wp + '_' + DM)
			

outputfile.Close()
print nfailed, 'failed fits'
print "The output ROOT file has been created: " + outputfile.GetName()
