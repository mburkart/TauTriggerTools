#!/usr/bin/env python

import ROOT
import argparse
import sys
import re
import numpy as np
from array import array

parser = argparse.ArgumentParser(description='Copmare two turnon curves.')
parser.add_argument('--input-a', required=True, type=str, help="the first input")
parser.add_argument('--input-b', required=True, type=str, help="the second input")
parser.add_argument('--labels', required=True, type=str, help="labels for both inputs")
parser.add_argument('--pattern', required=True, type=str, help="trigger name pattern")
parser.add_argument('--selection', required=True, type=str, help="Tau selection")
parser.add_argument('--output', required=True, type=str, help="output file")
parser.add_argument('--var', required=True, type=str, help="variable to draw")
args = parser.parse_args()

sys.path.insert(0, 'Common/python')
from AnalysisTypes import *
ROOT.gROOT.SetBatch(True)

def LoadTriggerDictionary(file):
    df_support = ROOT.RDataFrame('summary', file)
    summary = df_support.AsNumpy()
    trigger_index = np.array(summary['trigger_index'][0])
    trigger_pattern = np.array(summary['trigger_pattern'][0])
    trigger_dict = {}
    for n in range(len(trigger_pattern)):
        trigger_dict[trigger_pattern[n]] = trigger_index[n]
    return trigger_dict

def GetMatchedTriggers(trigger_dict, pattern):
    reg_ex = re.compile(pattern)
    matched = {}
    for name, pos in trigger_dict.items():
        if reg_ex.match(name) is not None:
            matched[name] = pos
    return matched

def ReportHLTPaths(hlt_paths, label):
    if len(hlt_paths) == 0:
        raise RuntimeError("No HLT path match the pattern for {}".format(label))
    line = 'HLT path for {}:'.format(label)
    for name in hlt_paths:
        line += ' {}'.format(name)
    print(line)

def CreateHistograms(input_file, selection_id, hlt_paths, label, var, hist_model, output_file):
    df = ROOT.RDataFrame('events', input_file)
    df = df.Filter('(tau_sel & {}) != 0'.format(selection_id))
    match_mask = 0
    for path_name, path_index in hlt_paths.items():
        match_mask = match_mask | (1 << path_index)
    hist_total = df.Histo1D(hist_model, var)
    hist_passed = df.Filter('(hlt_acceptAndMatch & {}) != 0'.format(match_mask)) \
                    .Histo1D(hist_model, var)
    eff = ROOT.TEfficiency(hist_passed.GetPtr(), hist_total.GetPtr())
    output_file.WriteTObject(hist_total.GetPtr(), label + '_total', 'Overwrite')
    output_file.WriteTObject(hist_passed.GetPtr(), label + '_passed', 'Overwrite')
    output_file.WriteTObject(eff, label + '_eff', 'Overwrite')
    return eff

selection_id = ParseEnum(TauSelection, args.selection)
print('Tau selection: {}'.format(args.selection))

labels = args.labels.split(',')

trigger_dict_a = LoadTriggerDictionary(args.input_a)
trigger_dict_b = LoadTriggerDictionary(args.input_b)

hlt_paths_a = GetMatchedTriggers(trigger_dict_a, args.pattern)
ReportHLTPaths(hlt_paths_a, labels[0])

hlt_paths_b = GetMatchedTriggers(trigger_dict_b, args.pattern)
ReportHLTPaths(hlt_paths_b, labels[1])

output_file = ROOT.TFile(args.output, 'RECREATE')

bins = np.arange(0, 100, step=10)
bins = np.append(bins, [100, 150, 200])
hist_model = ROOT.RDF.TH1DModel('tau_pt', '', len(bins) - 1, array('d', bins))

eff_a = CreateHistograms(args.input_a, selection_id, hlt_paths_a, labels[0], args.var, hist_model, output_file)
eff_b = CreateHistograms(args.input_b, selection_id, hlt_paths_b, labels[1], args.var, hist_model, output_file)

ratio = hist_model.GetHistogram().Clone('ratio')
for n in range(1, ratio.GetNbinsX() + 1):
    if eff_b.GetEfficiency(n) != 0:
        ratio.SetBinContent(n, eff_a.GetEfficiency(n) / eff_b.GetEfficiency(n))
output_file.WriteTObject(ratio, 'ratio', 'Overwrite')

output_file.Close()
