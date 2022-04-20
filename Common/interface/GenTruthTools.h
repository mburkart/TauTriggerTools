/*! Tools for working with MC generator truth.
This file is part of https://github.com/cms-tau-pog/TauTriggerTools. */

#pragma once

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "AnalysisTypes.h"

namespace analysis {

namespace gen_truth {

struct FinalState {
public:
    enum class ParticleType { visible, light_lepton, neutrino, gamma, charged_hadron, neutral_hadron };

    explicit FinalState(const reco::GenParticle& particle, const std::set<int>& pdg_to_exclude = {},
                        const std::set<const reco::GenParticle*>& particles_to_exclude = {});

    const std::set<const reco::GenParticle*>& getParticles(ParticleType type) { return particles[type]; }
    const LorentzVectorXYZ& getMomentum(ParticleType type) { return momentum[type]; }
    size_t count(ParticleType type) { return getParticles(type).size(); }

private:
    void findFinalStateParticles(const reco::GenParticle& particle, const std::set<int>& pdg_to_exclude,
                                 const std::set<const reco::GenParticle*>& particles_to_exclude);
    void addParticle(const reco::GenParticle& particle);

private:
    std::map<ParticleType, std::set<const reco::GenParticle*>> particles;
    std::map<ParticleType, LorentzVectorXYZ> momentum;
};

struct LeptonMatchResult {
    GenLeptonMatch match{GenLeptonMatch::NoMatch};
    const reco::GenParticle *gen_particle_firstCopy{nullptr}, *gen_particle_lastCopy{nullptr};
    std::set<const reco::GenParticle*> visible_daughters, visible_rad;
    LorentzVectorXYZ visible_p4, visible_rad_p4;
    unsigned n_charged_hadrons{0}, n_neutral_hadrons{0}, n_gammas{0}, n_gammas_rad{0};
};

const reco::GenParticle* FindTerminalCopy(const reco::GenParticle& genParticle, bool first);
bool FindLeptonGenMatch(const reco::GenParticle& particle, LeptonMatchResult& result,
                        const LorentzVectorM* ref_p4 = nullptr, double* best_match_dr2 = nullptr);

std::vector<LeptonMatchResult> CollectGenLeptons(const reco::GenParticleCollection& genParticles);
LeptonMatchResult LeptonGenMatch(const LorentzVectorM& p4, const reco::GenParticleCollection& genParticles);
LeptonMatchResult LeptonGenMatch(const LorentzVectorM& p4, const std::vector<LeptonMatchResult>& genLeptons);

float GetNumberOfPileUpInteractions(edm::Handle<std::vector<PileupSummaryInfo>>& pu_infos);

} // namespace gen_truth
} // namespace analysis
