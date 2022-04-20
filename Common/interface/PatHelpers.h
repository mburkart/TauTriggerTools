/*! Various utility functions.
This file is part of https://github.com/cms-tau-pog/TauTriggerTools. */

#pragma once

#include <Math/VectorUtil.h>
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TauTriggerTools/Common/interface/GenTruthTools.h"

namespace tau_trigger {

using namespace analysis;

double MuonIsolation(const pat::Muon& muon);

template<typename LVector1, typename LVector2>
double Calculate_MT(const LVector1& lepton_p4, const LVector2& met_p4)
{
    const double delta_phi = ROOT::Math::VectorUtil::DeltaPhi(lepton_p4, met_p4);
    return std::sqrt( 2.0 * lepton_p4.Pt() * met_p4.Pt() * ( 1.0 - std::cos(delta_phi) ) );
}

struct TauEntry {
    const pat::Tau* reco_tau{nullptr};
    gen_truth::LeptonMatchResult gen_tau;
    unsigned selection{0};
};

std::vector<TauEntry> CollectTaus(const LorentzVectorM& muon_p4, const pat::TauCollection& taus,
                                  const std::vector<gen_truth::LeptonMatchResult>& genLeptons, double deltaR2Thr);

bool PassBtagVeto(const LorentzVectorM& muon_p4, const LorentzVectorM& tau_p4, const pat::JetCollection& jets,
                  double btagThreshold, double deltaR2Thr);
gen_truth::LeptonMatchResult SelectGenLeg(const std::vector<gen_truth::LeptonMatchResult>& genLeptons, bool is_tau);

} // namespace tau_trigger
