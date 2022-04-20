/*! Various utility functions.
This file is part of https://github.com/cms-tau-pog/TauTriggerTools. */

#include "TauTriggerTools/Common/interface/PatHelpers.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "TauTriggerTools/Common/interface/AnalysisTypes.h"

namespace tau_trigger {

double MuonIsolation(const pat::Muon& muon)
{
    const double pfIso = muon.pfIsolationR04().sumChargedHadronPt
                         + std::max(0.0, muon.pfIsolationR04().sumNeutralHadronEt
                                    + muon.pfIsolationR04().sumPhotonEt - 0.5 * muon.pfIsolationR04().sumPUPt);
    return pfIso / muon.polarP4().pt();
}

std::vector<TauEntry> CollectTaus(const LorentzVectorM& muon_p4, const pat::TauCollection& taus,
                                  const std::vector<gen_truth::LeptonMatchResult>& genLeptons, double deltaR2Thr)
{
    static const std::string mvaIdName = "byIsolationMVArun2017v2DBoldDMwLTraw2017";
    static const std::string deepIdName = "byDeepTau2017v2p1VSjetraw";
    std::map<TauSelection, const pat::Tau*> best_tau;
    for(const auto& tau : taus) {
        auto leadChargedHadrCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());
        if(tau.polarP4().pt() > 18 && std::abs(tau.polarP4().eta()) < 2.3
                && leadChargedHadrCand && std::abs(leadChargedHadrCand->dz()) < 0.2
                && reco::deltaR2(muon_p4, tau.polarP4()) > deltaR2Thr) {
            const bool pass_mva_sel = tau.isTauIDAvailable(mvaIdName) && tau.tauID("againstMuonLoose3") > 0.5f;
            const bool pass_deep_sel = tau.isTauIDAvailable(deepIdName)
                && tau.tauID("byVVVLooseDeepTau2017v2p1VSe") > 0.5f
                && tau.tauID("byVLooseDeepTau2017v2p1VSmu") > 0.5f;
            if((pass_mva_sel || pass_deep_sel) && (!best_tau.count(TauSelection::pt)
                    || best_tau.at(TauSelection::pt)->polarP4().pt() < tau.polarP4().pt()))
                best_tau[TauSelection::pt] = &tau;
            if(pass_mva_sel && (!best_tau.count(TauSelection::MVA)
                    || best_tau.at(TauSelection::MVA)->tauID(mvaIdName) < tau.tauID(mvaIdName)))
                best_tau[TauSelection::MVA] = &tau;
            if(pass_deep_sel && (!best_tau.count(TauSelection::DeepTau)
                    || best_tau.at(TauSelection::DeepTau)->tauID(deepIdName) < tau.tauID(deepIdName)))
                best_tau[TauSelection::DeepTau] = &tau;
        }
    }
    std::map<const pat::Tau*, TauEntry> selected_taus;
    const gen_truth::LeptonMatchResult selected_gen_tau = SelectGenLeg(genLeptons, true);
    const bool has_selected_gen_tau = selected_gen_tau.match != GenLeptonMatch::NoMatch;
    bool selected_gen_tau_stored = false;
    for(const auto& entry : best_tau) {
        const pat::Tau* reco_tau = entry.second;
        if(!selected_taus.count(reco_tau)) {
            const auto gen_tau = gen_truth::LeptonGenMatch(reco_tau->polarP4(), genLeptons);
            const bool has_gen_tau = gen_tau.match != GenLeptonMatch::NoMatch;
            selected_taus[reco_tau] = TauEntry{reco_tau, gen_tau, 0};
            if(has_selected_gen_tau && has_gen_tau
                    && selected_gen_tau.gen_particle_firstCopy == gen_tau.gen_particle_firstCopy) {
                selected_gen_tau_stored = true;
                selected_taus[reco_tau].selection |= static_cast<unsigned>(TauSelection::gen);
            }
        }
        selected_taus[reco_tau].selection |= static_cast<unsigned>(entry.first);
    }
    if(has_selected_gen_tau && !selected_gen_tau_stored) {
        const pat::Tau* reco_tau = nullptr;
        for(const auto& tau : taus) {
            const auto gen_tau = gen_truth::LeptonGenMatch(tau.polarP4(), genLeptons);
            if(gen_tau.match != GenLeptonMatch::NoMatch
                    && gen_tau.gen_particle_firstCopy == selected_gen_tau.gen_particle_firstCopy) {
                reco_tau = &tau;
                break;
            }
        }
        if(selected_taus.count(reco_tau))
            throw exception("Inconsistency in CollectTaus algorithm.");
        selected_taus[reco_tau] = TauEntry{reco_tau, selected_gen_tau, static_cast<unsigned>(TauSelection::gen)};
    }

    std::vector<TauEntry> result;
    for(const auto& entry : selected_taus)
        result.push_back(entry.second);
    return result;
}

bool PassBtagVeto(const LorentzVectorM& muon_p4, const LorentzVectorM& tau_p4,
                  const pat::JetCollection& jets, double btagThreshold, double deltaR2Thr)
{
    if(btagThreshold > 0) {
        for(const pat::Jet& jet : jets) {
            const auto btag = jet.bDiscriminator("pfDeepFlavourJetTags:probb")
                              + jet.bDiscriminator("pfDeepFlavourJetTags:probbb")
                              + jet.bDiscriminator("pfDeepFlavourJetTags:problepb");
            if(reco::deltaR2(muon_p4, jet.polarP4()) > deltaR2Thr
                    && reco::deltaR2(tau_p4, jet.polarP4()) > deltaR2Thr
                    && jet.polarP4().pt() > 20 && std::abs(jet.polarP4().eta()) < 2.4
                    && btag > btagThreshold)
                return false;
        }
    }
    return true;
}

gen_truth::LeptonMatchResult SelectGenLeg(const std::vector<gen_truth::LeptonMatchResult>& genLeptons, bool is_tau)
{
    static const std::map<bool, std::set<GenLeptonMatch>> all_matches = {
        { true, { GenLeptonMatch::Tau } },
        { false, { GenLeptonMatch::Muon, GenLeptonMatch::TauMuon } },
    };
    const auto& matches = all_matches.at(is_tau);
    gen_truth::LeptonMatchResult leg;
    for(const auto& lepton : genLeptons) {
        if(matches.count(lepton.match) && (leg.match == GenLeptonMatch::NoMatch
                    || leg.visible_p4.pt() < lepton.visible_p4.pt())) {
            leg = lepton;
        }
    }
    return leg;
}

} // namespace tau_trigger
