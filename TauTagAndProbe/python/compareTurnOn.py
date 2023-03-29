#!/usr/bin/env python

import argparse
from array import array
import math
import numpy as np
import re
import sys
import ROOT

parser = argparse.ArgumentParser(description='Copmare turn-on curves.')
parser.add_argument('--input', required=True, type=str, nargs='+', help="input file")
parser.add_argument('--labels', required=True, type=str, help="labels for both inputs")
parser.add_argument('--pattern', required=True, type=str, help="trigger name pattern")
parser.add_argument('--selection', required=True, type=str, help="tau selection")
parser.add_argument('--output', required=True, type=str, help="output file prefix")
parser.add_argument('--vars', required=True, type=str, help="variables to draw")
parser.add_argument('--max-pt', required=False, type=float, default=None, help="max tau pt")
parser.add_argument('--max-gen-pt', required=False, type=float, default=None, help="max visible gen tau pt")
parser.add_argument('--min-hlt-pt', required=False, type=float, default=None,
                    help="minimal pt requiremet for the best matched HLT object to consider that the path is passed")
parser.add_argument('--deltaRThr', required=False, type=float, default=0.5, help="delta R threshold for HLT matching")
parser.add_argument('--min-l1-pt', required=False, type=float, default=None,
                    help="minimal pt requiremet for the best matched L1 object to consider that the path is passed")
args = parser.parse_args()

sys.path.insert(0, 'Common/python')
from AnalysisTypes import *
from AnalysisTools import *
import RootPlotting
import TriggerConfig
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
RootPlotting.ApplyDefaultGlobalStyle()

ccp_methods = '''
int FindBestMatchedHLTObject(float tau_eta, float tau_phi, ULong64_t match_mask, float deltaRThr,
                             const ROOT::VecOps::RVec<float>& hltObj_eta, const ROOT::VecOps::RVec<float>& hltObj_phi,
                             const ROOT::VecOps::RVec<ULong64_t>& hltObj_hasPathName,
                             const ROOT::VecOps::RVec<ULong64_t>& hltObj_hasFilters_2)
{
    int best_match_index = -1;
    float best_deltaR2 = std::pow(deltaRThr, 2);
    for(size_t n = 0; n < hltObj_eta.size(); ++n) {
        //if((match_mask & hltObj_hasPathName.at(n) & hltObj_hasFilters_2.at(n)) == 0) continue;
        if((match_mask & hltObj_hasPathName.at(n)) == 0) continue;
        const float deta = tau_eta - hltObj_eta.at(n);
        const float dphi = ROOT::Math::VectorUtil::Phi_mpi_pi(tau_phi - hltObj_phi.at(n));
        const float deltaR2 = std::pow(deta, 2) + std::pow(dphi, 2);
        if(deltaR2 >= best_deltaR2) continue;
        best_match_index = static_cast<int>(n);
        best_deltaR2 = deltaR2;
    }
    return best_match_index;
}
'''
ROOT.gInterpreter.Declare(ccp_methods)

def ReportHLTPaths(hlt_paths, label):
    if len(hlt_paths) == 0:
        raise RuntimeError("No HLT path match the pattern for {}".format(label))
    line = 'HLT path for {}:'.format(label)
    for name in hlt_paths:
        line += ' {}'.format(name)
    print(line)

def CreateBins(var_name, max_tau_pt, max_gen_pt):
    if var_name in [ 'tau_pt', 'tau_gen_vis_pt' ]:
        max_pt = 200
        if var_name == 'tau_pt' and max_tau_pt is not None:
            max_pt = max(max_tau_pt, 100)
        if var_name == 'tau_gen_vis_pt' and max_gen_pt is not None:
            max_pt = max(max_gen_pt, 100)
        bins = np.arange(10, 100, step=10)
        high_pt_bins = [ 100, 150, 200, 300, 400, 500, 650, 800, 1000 ]
        n = 0
        while n < len(high_pt_bins) and high_pt_bins[n] < max_pt:
            n += 1
        use_logx = max_pt > 200
        return np.append(bins, high_pt_bins[0:n+1]), use_logx, True
    elif var_name in [ 'tau_eta', 'tau_gen_vis_eta' ]:
        return np.linspace(-2.3, 2.3, 20), False, False
    elif var_name in [ 'npu', 'npv' ]:
        return np.linspace(0, 80, 20), False, False
    raise RuntimeError("Can't find binning for \"{}\"".format(var_name))

def GetTitle(var_name, axis_title=False):
    titles = {
        'tau_pt': ('#tau p_{T}', '#tau p_{T} (GeV)'),
        'tau_gen_vis_pt': ('visible gen #tau p_{T}', 'visible gen #tau p_{T} (GeV)'),
        'tau_eta': ('#tau #eta', ),
        'tau_gen_vis_eta': ('visible gen #tau #eta', ),
        'npv': ('Number of reconstructed PV', '# reco PV'),
        'npu': ('Number of generated PV', '# gen PV'),
    }
    if var_name in titles:
        index = min(int(axis_title), len(titles[var_name]) - 1)
        return titles[var_name][index]
    return var_name

def CreateHistograms(input_file, selection_id, hlt_paths, label, vars, hist_models, output_file):
    df = ROOT.RDataFrame('events', input_file)
    df = df.Filter('(tau_sel & {}) != 0 && abs(tau_eta) < 2.3'.format(selection_id))
    if args.max_gen_pt is not None:
        df = df.Filter('tau_gen_vis_pt < {}'.format(args.max_gen_pt))
    match_mask = 0
    for path_name, path_index in hlt_paths.items():
        match_mask = match_mask | (1 << path_index)
    if selection_id == TauSelection.gen:
        tau_eta_branch = 'tau_gen_vis_eta'
        tau_phi_branch = 'tau_gen_vis_phi'
    else:
        tau_eta_branch = 'tau_eta'
        tau_phi_branch = 'tau_phi'
    df_passed = df.Filter('(hlt_accept & {}ULL) != 0'.format(match_mask)) \
                  .Define('matched_hlt_idx',
                          '''FindBestMatchedHLTObject({}, {}, {}ULL, {}, hltObj_eta, hltObj_phi,
                                                      hltObj_hasPathName, hltObj_hasFilters_2)''' \
                          .format(tau_eta_branch, tau_phi_branch, match_mask, args.deltaRThr)) \
                  .Filter('matched_hlt_idx >= 0')

    if args.min_hlt_pt is not None:
        df_passed = df_passed.Filter('hltObj_pt.at(matched_hlt_idx) > {}'.format(args.min_hlt_pt))
    if args.min_l1_pt is not None:
        df_passed = df_passed.Filter('l1Tau_pt > {}'.format(args.min_l1_pt))
    hist_total = {}
    hist_passed = {}
    for var_id in range(len(vars)):
        var = vars[var_id]
        hist_total[var] = df.Histo1D(hist_models[var], var)
        hist_passed[var] = df_passed.Histo1D(hist_models[var], var)

    eff = {}
    for var_id in range(len(vars)):
        var = vars[var_id]
        eff[var] = ROOT.TEfficiency(hist_passed[var].GetPtr(), hist_total[var].GetPtr())
        eff[var].SetStatisticOption(ROOT.TEfficiency.kFCP)
        output_file.WriteTObject(hist_total[var].GetPtr(), '{}_total_{}'.format(label, var), 'Overwrite')
        output_file.WriteTObject(hist_passed[var].GetPtr(), '{}_passed_{}'.format(label, var), 'Overwrite')
        output_file.WriteTObject(eff[var], '{}_eff_{}'.format(label, var), 'Overwrite')
    return hist_passed, hist_total, eff

selection_id = ParseEnum(TauSelection, args.selection)
print('Tau selection: {}'.format(args.selection))

n_inputs = len(args.input)
labels = args.labels.split(',')
vars = args.vars.split(',')

if len(labels) != n_inputs:
    raise RuntimeError("Inconsitent number of inputs = {} and number of labels = {}".format(n_inputs, len(labels)))

trigger_dict = [None] * n_inputs
hlt_paths = [None] * n_inputs
for input_id in range(n_inputs):
    trigger_dict[input_id] = TriggerConfig.LoadTriggerDictionary(args.input[input_id])
    hlt_paths[input_id] = TriggerConfig.GetMatchedTriggers(trigger_dict[input_id], args.pattern)
    ReportHLTPaths(hlt_paths[input_id], labels[input_id])

output_file = ROOT.TFile(args.output + '.root', 'RECREATE')

bins = {}
x_scales = {}
divide_by_bw = {}
hist_models = {}
for var_id in range(len(vars)):
    var = vars[var_id]
    bins[var], x_scales[var], divide_by_bw[var] = CreateBins(var, args.max_pt, args.max_gen_pt)
    hist_models[var] = ROOT.RDF.TH1DModel(var, var, len(bins[var]) - 1, array('d', bins[var]))

hist_passed = [None] * n_inputs
hist_total = [None] * n_inputs
eff = [None] * n_inputs

for input_id in range(n_inputs):
    hist_passed[input_id], hist_total[input_id], eff[input_id] = \
        CreateHistograms(args.input[input_id], selection_id, hlt_paths[input_id], labels[input_id], vars, hist_models,
                         output_file)

colors = [ROOT.kBlue, ROOT.kRed]
canvas = RootPlotting.CreateCanvas()

target_names = [ 'passed', 'total', 'efficiency' ]
is_efficiency = [ False, False, True]
targets = [ hist_passed, hist_total, eff ]

n_plots = len(vars) * len(targets)
plot_id = 0
for var_id in range(len(vars)):
    var = vars[var_id]
    for target_id in range(len(targets)):
        ratio_graphs = {}
        ref_hist = hist_models[var].GetHistogram()
        ratio_ref_hist = ref_hist.Clone()
        y_title = 'Efficiency' if is_efficiency[target_id] else 'arb.'
        curves = [None] * n_inputs
        for input_id in range(n_inputs):
            curve = targets[target_id][input_id][var]
            if 'RResultPtr' in str(type(curve)):
                curve = curve.GetPtr()
            if not is_efficiency[target_id]:
                curve = curve.Clone()
                curve.Scale(1. / curve.Integral())
                if divide_by_bw[var]:
                    RootPlotting.DivideByBinWidth(curve)
            curves[input_id] = curve
        if is_efficiency[target_id]:
            y_min, y_max = (0, 1)
        else:
            y_min = 0
            _, y_max = RootPlotting.GetYRange(curves)

        title = '{}: {}'.format(GetTitle(var, False), target_names[target_id])
        plain_title = '{}_{}'.format(var, target_names[target_id])
        main_pad, ratio_pad, title_controls = RootPlotting.CreateTwoPadLayout(canvas, ref_hist, ratio_ref_hist,
                                                                              log_x=x_scales[var], title=title)
        RootPlotting.ApplyAxisSetup(ref_hist, ratio_ref_hist, x_title=GetTitle(var, True), y_title=y_title,
                                    ratio_y_title='Ratio', y_range=(y_min, y_max * 1.1), max_ratio=1.5)
        legend = RootPlotting.CreateLegend(pos=(0.18, 0.78), size=(0.2, 0.15))
        for input_id in range(n_inputs):
            curve = curves[input_id]
            curve.Draw('SAME')
            RootPlotting.ApplyDefaultLineStyle(curve, colors[input_id])
            legend.AddEntry(curve, labels[input_id], 'PLE')

            if input_id < n_inputs - 1:
                if is_efficiency[target_id]:
                    ratio_graphs[var] = RootPlotting.CreateEfficiencyRatioGraph(hist_passed[input_id][var],
                                                                                hist_total[input_id][var],
                                                                                hist_passed[-1][var],
                                                                                hist_total[-1][var])
                else:
                    ratio_hist = curve.Clone()
                    ratio_hist.Divide(curves[-1])
                    ratio_graphs[var] = RootPlotting.HistogramToGraph(ratio_hist)
                if ratio_graphs[var]:
                    output_file.WriteTObject(ratio_graphs[var],
                                             '{}_ratio_{}_{}'.format(var, labels[input_id], labels[-1]),
                                             'Overwrite')
                    ratio_pad.cd()
                    ratio_color = colors[input_id] if n_inputs > 2 else ROOT.kBlack
                    RootPlotting.ApplyDefaultLineStyle(ratio_graphs[var], ratio_color)
                    ratio_graphs[var].Draw("0PE SAME")
                    main_pad.cd()
        legend.Draw()

        canvas.Update()
        output_file.WriteTObject(canvas, '{}_canvas'.format(plain_title), 'Overwrite')
        RootPlotting.PrintAndClear(canvas, args.output + '.pdf', plain_title, plot_id, n_plots, [main_pad, ratio_pad])
        plot_id += 1
output_file.Close()
