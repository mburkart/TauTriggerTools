/*! Definition of trigger results.
This file is part of https://github.com/cms-tau-pog/TauTriggerTools. */

#pragma once

#include <bitset>
#include <boost/regex.hpp>

#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisTypes.h"

namespace tau_trigger {

struct TriggerLeg {
    analysis::LegType type;
    std::vector<std::string> filters;
};

struct TriggerDescriptor {
    std::string path;
    boost::regex regex;
    int global_index{-1};
    std::vector<TriggerLeg> legs;
    bool is_tag{false};
    unsigned type_mask{0};
};

unsigned GetTriggerObjectTypes(const pat::TriggerObjectStandAlone& triggerObject);
const l1t::Tau* MatchL1Taus(const analysis::LorentzVectorM& ref_p4, const BXVector<l1t::Tau>& l1Taus, double deltaR2Thr,
                           int bx_value);


using TriggerBitsContainer = unsigned long long;
constexpr size_t MaxNumberOfTriggers = std::numeric_limits<TriggerBitsContainer>::digits;
using TriggerResults = std::bitset<MaxNumberOfTriggers>;

struct TriggerObjectMatchResult {
    size_t hltObjIndex;
    unsigned objType;

    TriggerResults hasPathName, isBestMatch;
    std::set<size_t> descIndices;
    std::vector<std::string> filters;
};

struct FullTriggerResults {
    TriggerResults accept, acceptAndMatch;
    std::map<size_t, TriggerObjectMatchResult> matchResults;
};

class TriggerDescriptorCollection {
public:
    TriggerDescriptorCollection(const edm::VParameterSet& trig_pset);
    const std::vector<TriggerDescriptor>& getDescriptors() const { return descs; }
    const TriggerDescriptor& at(size_t n) const { return descs.at(n); }
    size_t getIndex(const std::string& path) const { return desc_indices.at(path); }
    size_t size() const { return descs.size(); }
    const std::set<size_t>& getTagDescriptorsIndices() const { return tag_desc_indices; }

    FullTriggerResults matchTriggerObjects(const edm::TriggerResults& triggerResults,
                                           const pat::TriggerObjectStandAloneCollection& triggerObjects,
                                           const analysis::LorentzVectorM& ref_p4,
                                           const std::vector<std::string>& triggerNames, double deltaR2Thr,
                                           bool include_tag_paths, bool include_nontag_paths);

    void updateGlobalIndices(const std::vector<std::string>& triggerNames);

private:
    std::vector<TriggerDescriptor> descs;
    std::map<std::string, size_t> desc_indices;
    std::set<size_t> tag_desc_indices;
};

}
